
import numpy as np
import torch
from collections import OrderedDict
import torch.nn as nn


def build_image_matrix(images, n_rows, n_cols):
    image_shape = images.shape[1:]

    image_matrix = np.zeros((n_rows * image_shape[0], n_cols * image_shape[1], 3), dtype=np.uint8)
    for i in range(n_cols):
        for j in range(n_rows):
            image_matrix[j * image_shape[0] : (j + 1) * image_shape[0], i * image_shape[1] : (i + 1) * image_shape[1]] = images[j * n_cols + i]

    return image_matrix


def adjust_learning_rate(optimizers, decay, number_decay, base_lr):
    lr = base_lr * (decay ** number_decay)
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


class RunningStatistics(object):
    def __init__(self):
        self.losses = OrderedDict()

    def add(self, key, value):
        if key not in self.losses:
            self.losses[key] = []
        self.losses[key].append(value)

    def means(self):
        return OrderedDict([
            (k, np.mean(v)) for k, v in self.losses.items() if len(v) > 0
        ])

    def reset(self):
        for key in self.losses.keys():
            self.losses[key] = []


def recover_images(x):
    x = x.cpu().numpy()
    x = (x + 1.0) * (255.0 / 2.0)
    x = np.clip(x, 0, 255)  # Avoid artifacts due to slight under/overflow
    x = x.astype(np.uint8)
    if len(x.shape) == 4:
        x = np.transpose(x, [0, 2, 3, 1])  # CHW to HWC
        x = x[:, :, :, ::-1]  # RGB to BGR for OpenCV
    else:
        x = np.transpose(x, [1, 2, 0])  # CHW to HWC
        x = x[:, :, ::-1]  # RGB to BGR for OpenCV
    return x


def send_data_dict_to_gpu(data1, data2, device):
    data = {}
    for k, v in data1.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(device, non_blocking=True)
    for k, v in data2.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(device, non_blocking=True)
    return data


class convert_dict(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [convert_dict(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, convert_dict(b) if isinstance(b, dict) else b)


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) ==0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def pitchyaw_to_vector(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def vector_to_pitchyaw(vectors):
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


radians_to_degrees = 180.0 / np.pi


def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = pitchyaw_to_vector(a)
    b = pitchyaw_to_vector(b)

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * radians_to_degrees

def get_act(activation_type):
    if activation_type == 'relu':
        return nn.ReLU
    elif activation_type == 'leaky_relu':
        return nn.LeakyReLU
    elif activation_type == 'elu':
        return nn.ELU
    elif activation_type == 'selu':
        return nn.SELU
    elif activation_type == 'tanh':
        return nn.Tanh
    elif activation_type == 'sigmoid':
        return nn.Sigmoid
    else:
        raise ValueError('Type of activation not valid')


def get_norm(norm_type):
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'layer':
        return nn.LayerNorm
    elif norm_type == 'instance':
        return nn.InstanceNorm2d
    elif norm_type == 'sync':
        return nn.SyncBatchNorm
    else:
        raise ValueError('Type of norm layer not valid')
