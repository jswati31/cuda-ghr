
import os
import torch
import numpy as np
from torch.utils.data import Dataset

import cv2 as cv
import h5py


class HDFDataset(Dataset):
    """Dataset from HDF5 archives formed of 'groups' of specific persons."""

    def __init__(self, hdf_file_path,
                 key_data='',
                 prefixes=None,
                 pick_exactly_per_person=None,
                 pick_at_least_per_person=None,
                 get_2nd_sample=False):
        assert os.path.isfile(hdf_file_path)
        self.pick_exactly_per_person = pick_exactly_per_person
        self.hdf_path = hdf_file_path
        self.key_data = key_data
        self.get_2nd_sample = get_2nd_sample
        self.hdf = h5py.File(hdf_file_path, 'r', libver='latest', swmr=True)

        with h5py.File(self.hdf_path, 'r', libver='latest', swmr=True) as h5f:
            hdf_keys = sorted(list(h5f.keys()))
            self.prefixes = hdf_keys if prefixes is None else prefixes
            if pick_exactly_per_person is not None:
                assert pick_at_least_per_person is None
                # Pick exactly x many entries from front of group
                self.prefixes = [
                    k for k in self.prefixes if k in h5f
                    and len(next(iter(h5f[k].values()))) >= pick_exactly_per_person
                ]
                self.index_to_query = sum([
                    [(prefix, i) for i in range(pick_exactly_per_person)]
                    for prefix in self.prefixes
                ], [])
            elif pick_at_least_per_person is not None:
                assert pick_exactly_per_person is None
                # Pick people for which there exists at least x many entries
                self.prefixes = [
                    k for k in self.prefixes if k in h5f
                    and len(next(iter(h5f[k].values()))) >= pick_at_least_per_person
                ]
                self.index_to_query = sum([
                    [(prefix, i) for i in range(len(next(iter(h5f[prefix].values()))))]
                    for prefix in self.prefixes
                ], [])
            else:
                # Pick all entries of person
                self.prefixes = [  # to address erroneous inputs
                    k for k in self.prefixes if k in h5f
                    and len(next(iter(h5f[k].values()))) > 0
                ]
                self.index_to_query = sum([
                    [(prefix, i) for i in range(len(next(iter(h5f[prefix].values()))))]
                    for prefix in self.prefixes
                ], [])

    def __len__(self):
        return len(self.index_to_query)

    def close_hdf(self):
        if self.hdf is not None:
            self.hdf.close()
            self.hdf = None

    def preprocess_image(self, image):
        ycrcb = cv.cvtColor(image, cv.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv.equalizeHist(ycrcb[:, :, 0])
        image = cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB)
        image = np.transpose(image, [2, 0, 1])  # Colour image
        image = 2.0 * image / 255.0 - 1
        return image

    def preprocess_entry(self, entry):
        for key, val in entry.items():
            if isinstance(val, np.ndarray):
                entry[key] = torch.from_numpy(val.astype(np.float32))
            elif isinstance(val, int):
                # NOTE: maybe ints should be signed and 32-bits sometimes
                entry[key] = torch.tensor(val, dtype=torch.int16, requires_grad=False)
        return entry

    def __getitem__(self, idx):
        if self.hdf is None:  # Need to lazy-open this to avoid read error
            self.hdf = h5py.File(self.hdf_path, 'r', libver='latest', swmr=True)

        # Pick a entry
        key_a, idx_a = self.index_to_query[idx]
        group_a = self.hdf[key_a]

        def retrieve(group, index):
            eyes = self.preprocess_image(group['pixels'][index, :])
            g = group['labels'][index, :2]
            h = group['labels'][index, 2:4]
            return eyes, g, h

        # Grab entry data
        eyes_a, g_a, h_a = retrieve(group_a, idx_a)
        entry = {
            self.key_data + '_key': key_a,
            self.key_data + '_key_index': self.prefixes.index(key_a),
            self.key_data + '_image': eyes_a,
            self.key_data + '_gaze': g_a,
            self.key_data + '_head': h_a,
        }

        if self.get_2nd_sample:
            # Grab 2nd entry from same person if needed (only needed for evaluation)
            group_b = group_a
            all_indices = list(range(len(next(iter(group_a.values())))))
            all_indices_but_a = np.delete(all_indices, idx_a)
            idx_b = np.random.choice(all_indices_but_a)

            eyes_b, g_b, h_b = retrieve(group_b, idx_b)
            entry['image_b'] = eyes_b
            entry['gaze_b'] = g_b
            entry['head_b'] = h_b

        return self.preprocess_entry(entry)
