

from torch.utils.data import DataLoader
import json
import losses
from tqdm import tqdm
from utils import RunningStatistics
import argparse
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from nets import Controller
from argparse import Namespace
import cv2
import h5py
import lpips
import random
import logging
import torch.nn.functional as F
from collections import OrderedDict
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


lpipsmodel = lpips.LPIPS(net='alex').to(device)
for param in lpipsmodel.parameters():
    param.requires_grad = False


class HDFDataset(Dataset):
    """Dataset from HDF5 archives formed of 'groups' of specific persons."""

    def __init__(self, hdf_file_path,
                 prefixes=None,
                 is_bgr=False,
                 get_2nd_sample=False,
                 pick_at_least_per_person=None,
                 num_labeled_samples=None,
                 sample_target_label=False,
                 ):
        assert os.path.isfile(hdf_file_path)
        self.get_2nd_sample = get_2nd_sample
        self.hdf_path = hdf_file_path
        self.hdf = None
        self.is_bgr = is_bgr
        self.sample_target_label = sample_target_label

        with h5py.File(self.hdf_path, 'r', libver='latest', swmr=True) as h5f:
            hdf_keys = sorted(list(h5f.keys()))
            if prefixes is None:
                self.prefixes = hdf_keys
            else:
                self.prefixes = [k for k in prefixes if k in h5f]
            if pick_at_least_per_person is not None:
                self.prefixes = [k for k in self.prefixes if k in h5f and len(next(iter(h5f[k].values()))) >=
                            pick_at_least_per_person]
            self.index_to_query = sum([[(prefix, i) for i in range(len(next(iter(h5f[prefix].values()))))]
                                       for prefix in self.prefixes], [])
            if num_labeled_samples is not None:
                # randomly pick labeled samples for semi-supervised training
                ra = list(range(len(self.index_to_query)))
                random.seed(0)
                random.shuffle(ra)
                ra = ra[:num_labeled_samples]
                list.sort(ra)
                self.index_to_query = [self.index_to_query[i] for i in ra]

            # calculate kernel density of gaze and head pose, for generating new redirected samples
            if sample_target_label:
                if num_labeled_samples is not None:
                    sample = []
                    old_key = -1
                    for key, idx in self.index_to_query:
                        if old_key != key:
                            group = h5f[key]
                        sample.append(group['labels'][idx, :4])
                    sample = np.asarray(sample, dtype=np.float32)
                else:
                    # can calculate faster if load by group
                    sample = None
                    for key in self.prefixes:
                        group = h5f[key]
                        if sample is None:
                            sample = group['labels'][:, :4]
                        else:
                            sample = np.concatenate([sample, group['labels'][:, :4]], axis=0)
                sample = sample.transpose()
                from scipy import stats
                self.kernel = stats.gaussian_kde(sample)
                logging.info("Finished calculating kernel density for gaze and head angles")
                # Sample new gaze and head pose angles
                new_samples = self.kernel.resample(len(self.index_to_query))
                self.gaze = new_samples[:2, :].transpose()
                self.head = new_samples[2:4, :].transpose()
                self.index_of_sample = 0

    def __len__(self):
        return len(self.index_to_query)

    def close_hdf(self):
        if self.hdf is not None:
            self.hdf.close()
            self.hdf = None

    def preprocess_image(self, image):
        if self.is_bgr:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        image = np.transpose(image, [2, 0, 1])  # Colour image
        image = 2.0 * image / 255.0 - 1
        return image

    def preprocess_entry(self, entry):
        for key, val in entry.items():
            if isinstance(val, np.ndarray):
                entry[key] = torch.from_numpy(val.astype(np.float32))
            elif isinstance(val, int):
                # NOTE: maybe ints should be signed and 32-bits sometimes
                entry[key] = torch.tensor(val, dtype=torch.long, requires_grad=False)
        return entry

    def __getitem__(self, idx):
        random.seed(idx)
        if self.hdf is None:  # Need to lazy-open this to avoid read error
            self.hdf = h5py.File(self.hdf_path, 'r', libver='latest', swmr=True)

        # Pick entry a and b from same person
        key_a, idx_a = self.index_to_query[idx]
        group_a = self.hdf[key_a]
        group_b = group_a

        def retrieve(group, index):
            eyes = self.preprocess_image(group['pixels'][index, :])
            g = group['labels'][index, :2]
            h = group['labels'][index, 2:4]
            return eyes, g, h
        # Grab 1st (input) entry
        eyes_a, g_a, h_a = retrieve(group_a, idx_a)
        entry = {
            'key': key_a,
            'image_a': eyes_a,
            'gaze_a': g_a,
            'head_a': h_a,
        }
        if self.sample_target_label:
            entry['gaze_b_r'] = self.gaze[self.index_of_sample]
            entry['head_b_r'] = self.head[self.index_of_sample]
            self.index_of_sample += 1
        if self.get_2nd_sample:
            all_indices = list(range(len(next(iter(group_a.values())))))
            if len(all_indices) == 1:
                # If there is only one sample for this person, just return the same sample.
                idx_b = idx_a
            else:
                all_indices_but_a = np.delete(all_indices, idx_a)
                idx_b = np.random.choice(all_indices_but_a)
            # Grab 2nd entry from same person
            eyes_b, g_b, h_b = retrieve(group_b, idx_b)
            entry['image_b'] = eyes_b
            entry['gaze_b'] = g_b
            entry['head_b'] = h_b
        return self.preprocess_entry(entry)


def send_data_dict_to_gpu(data):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(device, non_blocking=True)
    return data


def lpips_eval(test_dataloader, network):
    
    lpips_dict = OrderedDict()

    lpips_metrics = []

    for _, tgt_data in tqdm(enumerate(test_dataloader)):
        torch.cuda.empty_cache()

        data = send_data_dict_to_gpu(tgt_data)

        with torch.no_grad():

            # source gaze embedding
            s_gaze_embedding = network.gaze_latent_encoder(data['gaze_b'])

            # tgt appearance embedding
            t_app_embedding = network.image_encoder(data['image_a'])

            both_swapped_image = network.generator([torch.cat((t_app_embedding, s_gaze_embedding), dim=-1),
                                                   F.pad(input=data['head_b'], pad=(0, 1), mode='constant',
                                                         value=0)])[0]
    
            lp = lpipsmodel(data['image_b'], both_swapped_image)
            lpips_metrics.append(lp.cpu().numpy())

    lpips_metrics = np.concatenate(lpips_metrics, axis=0)

    lpips_dict['lpips'] = np.mean(lpips_metrics)
    
    return lpips_dict


def redir_eval(test_dataloader, src_dataloader, network):

    test_losses = RunningStatistics()
    src_data_iterator = iter(src_dataloader)

    with torch.no_grad():

        for _, data in tqdm(enumerate(test_dataloader)):
            torch.cuda.empty_cache()

            try:
                src_input = next(src_data_iterator)
            except StopIteration:
                src_data_iterator = iter(src_dataloader)
                src_input = next(src_data_iterator)

            data = send_data_dict_to_gpu(data)
            src_input = send_data_dict_to_gpu(src_input)
            
            # pseudo labels for tgt image
            psuedo_label_g, psuedo_label_h = network.task_net(data['image_a'])

            # source gaze embeddings
            s_gaze_embedding = network.gaze_latent_encoder(src_input['gaze_b_r'])
            
            # tgt gaze and appearance embeddings
            t_app_embedding = network.image_encoder(data['image_a'])
            t_gaze_embedding = network.gaze_latent_encoder(psuedo_label_g)

            gaze_swapped_image = network.generator([torch.cat((t_app_embedding, s_gaze_embedding), dim=-1),
                                                   F.pad(input=psuedo_label_h, pad=(0, 1), mode='constant', value=0)])[0]
    
            head_swapped_image = network.generator([torch.cat((t_app_embedding, t_gaze_embedding), dim=-1),
                                                   F.pad(input=src_input['head_b_r'], pad=(0, 1), mode='constant', value=0)])[0]

            _, redir_head = network.task_net(head_swapped_image)
            redir_gaze, _ = network.task_net(gaze_swapped_image)

            gaze_error = losses.gaze_angular_loss(src_input['gaze_b_r'], redir_gaze)
            head_error = losses.gaze_angular_loss(src_input['head_b_r'], redir_head)
          
            test_losses.add('gaze_redirection_errors', gaze_error.detach().cpu().numpy())
            test_losses.add('head_redirection_errors', head_error.detach().cpu().numpy())

    return test_losses.means()


def disent_eval(test_dataloader, src_dataloader, network):

    test_losses = RunningStatistics()
    src_data_iterator = iter(src_dataloader)

    with torch.no_grad():

        for _, data in tqdm(enumerate(test_dataloader)):
            torch.cuda.empty_cache()

            try:
                src_input = next(src_data_iterator)
            except StopIteration:
                src_data_iterator = iter(src_dataloader)
                src_input = next(src_data_iterator)

            data = send_data_dict_to_gpu(data)
            src_input = send_data_dict_to_gpu(src_input)
            
            # pseudo labels
            psuedo_label_g, psuedo_label_h = network.task_net(data['image_a'])

            # tgt image
            t_app_embedding = network.image_encoder(data['image_a'])
            t_gaze_embedding = network.gaze_latent_encoder(psuedo_label_g)

            recons_img = network.generator([torch.cat((t_app_embedding, t_gaze_embedding), dim=-1),
                                            F.pad(input=psuedo_label_h, pad=(0, 1), mode='constant', value=0)])[0]
            comp_g, comp_h = network.task_net(recons_img)

            for type in ['gaze', 'head']:
                if type == 'head':

                    head_swapped_image = network.generator([torch.cat((t_app_embedding, t_gaze_embedding), dim=-1),
                                                   F.pad(input=src_input['head_b_r'], pad=(0, 1), mode='constant',
                                                         value=0)])[0]
    
                    psuedo_label_random_gaze, _ = network.task_net(head_swapped_image)
                    gaze_error = losses.gaze_angular_loss(comp_g, psuedo_label_random_gaze)
                    test_losses.add('head_to_gaze', gaze_error.cpu().numpy())

                else:
                    s_gaze_embedding = network.gaze_latent_encoder(src_input['gaze_b_r'])
                    gaze_swapped_image = network.generator([torch.cat((t_app_embedding, s_gaze_embedding), dim=-1),
                                                   F.pad(input=psuedo_label_h, pad=(0, 1), mode='constant',
                                                         value=0)])[0]
    
                    _, psuedo_label_random_head = network.task_net(gaze_swapped_image)
                    head_error = losses.gaze_angular_loss(comp_h, psuedo_label_random_head)

                    test_losses.add('gaze_to_head', head_error.cpu().numpy())

    return test_losses.means()


def get_data(cfg):
    
    # Load GazeCapture prefixes with train/val/test split spec.
    with open('./gazecapture_split.json', 'r') as f:
        all_gc_prefixes = json.load(f)

    train_prefixes = all_gc_prefixes['train']
    src_dataset = HDFDataset(hdf_file_path=cfg.gazecapture_file,
                               prefixes=train_prefixes, sample_target_label=True)

    src_dataloader = DataLoader(src_dataset,
                              batch_size=32,
                              drop_last=True,
                              num_workers=1,
                              pin_memory=True,
                                shuffle=False)

    if args.columbia:
        # unseen
        if cfg.test_people == 6:
            _keys = ['{:04d}'.format(i) for i in range(51, 57)]
        # seen
        elif cfg.test_people == 50:
            _keys = ['{:04d}'.format(i) for i in range(1, 51)]
        # all
        else:
            _keys = ['{:04d}'.format(i) for i in range(1, 57)]

        test_dataset = HDFDataset(hdf_file_path=cfg.columbia_file,
                                   prefixes=_keys,
                                   get_2nd_sample=True,
                                   )
    else:
        # unseen
        if cfg.test_people == 4:
            _keys = ['p{:02d}'.format(i) for i in range(11, 15)]
        # seen
        elif cfg.test_people == 11:
            _keys = ['p{:02d}'.format(i) for i in range(11)]
        # all
        else:
            _keys = ['p{:02d}'.format(i) for i in range(15)]

        test_dataset = HDFDataset(hdf_file_path=cfg.mpiigaze_file,
                                   prefixes=_keys,
                                   get_2nd_sample=True,
                                   )
    print('Testing on ', _keys)

    test_dataloader = DataLoader(test_dataset,
                              batch_size=32,
                              drop_last=True,
                              num_workers=1,
                              pin_memory=True, shuffle=False)

    return src_dataloader, test_dataloader


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Evaluation code.')
    parser.add_argument('--model_path', type=str, help='Path to test model')
    parser.add_argument('--config_json', type=str, help='Path to config file')
    parser.add_argument('--test_people', type=int, default=15, help='test only on mpii excluded people')
    parser.add_argument('--columbia', action='store_true', help='test on columbia if true')

    args = parser.parse_args()
    
    config = json.load(open(args.config_json))
    config = Namespace(**config)
    config.test_people = args.test_people
    
    network = Controller(config, device).to(device)
    network.push_modules_to_multi_gpu()

    network.load_model(args.model_path)

    network.eval_mode_on()
    
    src_dataloader, tgt_dataloader = get_data(config)
    
    lpips_metrics_dict = lpips_eval(tgt_dataloader, network)
    redirect_metrics_dict = redir_eval(tgt_dataloader, src_dataloader, network)
    disentangled_metrics_dict = disent_eval(tgt_dataloader, src_dataloader, network)
    
    print('Metrics: %s %s %s' % (
        ', '.join(['%s: %.6f' % v for v in lpips_metrics_dict.items()]),
        ', '.join(['%s: %.6f' % v for v in redirect_metrics_dict.items()]), 
        ', '.join(['%s: %.6f' % v for v in disentangled_metrics_dict.items()])
    ))

