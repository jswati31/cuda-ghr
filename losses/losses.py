import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .vgg import Vgg16
from torchvision import transforms


def discriminator_loss(real, fake):
    GANLoss = nn.BCEWithLogitsLoss(reduction='mean')
    real_size = list(real.size())
    fake_size = list(fake.size())
    device = real.get_device()
    real_label = torch.zeros(real_size, dtype=torch.float32).to(device)
    fake_label = torch.ones(fake_size, dtype=torch.float32).to(device)

    discriminator_loss = (GANLoss(fake, fake_label) + GANLoss(real, real_label)) / 2

    return discriminator_loss


def generator_loss(fake):
    GANLoss = nn.BCEWithLogitsLoss(reduction='mean')
    fake_size = list(fake.size())
    device = fake.get_device()
    fake_label = torch.zeros(fake_size, dtype=torch.float32).to(device)
    return GANLoss(fake, fake_label)


def reconstruction_l1_loss(x, x_hat):
    loss_fn = nn.L1Loss(reduction='mean')
    return loss_fn(x, x_hat)


def nn_angular_distance(a, b):
    sim = F.cosine_similarity(a, b, eps=1e-6)
    sim = F.hardtanh(sim, -1.0, 1.0)
    return torch.acos(sim) * (180 / np.pi)


def pitchyaw_to_vector(pitchyaws):
    sin = torch.sin(pitchyaws)
    cos = torch.cos(pitchyaws)
    return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], 1)


def gaze_angular_loss(y, y_hat):
    y = pitchyaw_to_vector(y)
    y_hat = pitchyaw_to_vector(y_hat)
    loss = nn_angular_distance(y, y_hat)
    return torch.mean(loss)


class PerceptualLoss():
    def __init__(self, device):

        self.mse_loss = torch.nn.MSELoss()

        self._pretrained_model = Vgg16().to(device)

    def loss(self, predicted, data):
        terms = self._loss_terms(predicted, data)

        sum_of_terms = 0
        for term in terms:
            sum_of_terms += term

        return sum_of_terms

    def _preprocess_input(self, input_img):
        preprocessed_input = input_img + 1

        _transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # normalize with ImageNet values
        ])

        preprocessed_input = _transform(preprocessed_input)

        return preprocessed_input

    def _loss_terms(self, predicted, data):

        preprocessed_predicted = self._preprocess_input(predicted)
        preprocessed_data = self._preprocess_input(data)

        all_activations_predicted = self._pretrained_model(preprocessed_predicted)
        all_activations_data = self._pretrained_model(preprocessed_data)

        loss_terms = []
        for activations_predicted, activations_data in zip(all_activations_predicted, all_activations_data):
            loss_terms.append(torch.nn.MSELoss(reduction='mean')(activations_predicted, activations_data))

        return loss_terms
