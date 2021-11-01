import torch.nn as nn
from nets import Encoder, HologanGenerator, Discriminator, MLPNetwork, GazeHeadResNet
import torch
import os
from utils import get_act, get_norm


class Controller(nn.Module):

    def __init__(self, config, device):
        super(Controller, self).__init__()

        self.config = config
        self.device = device

        cat_latent_dim = config.z_dim_gaze + config.z_dim_app

        self.image_encoder = Encoder(z_dim_app=config.z_dim_app,
                                     num_blocks=config.num_blocks,
                                     normalization_fn=get_norm(config.encoder_norm),
                                     activation_fn=get_act(config.encoder_act)).to(device)

        self.generator = HologanGenerator(latent_dim=cat_latent_dim,
                                          output_shape=(config.input_img_h, config.input_img_w, 3),
                                          gen_output_activation=nn.Tanh).to(device)

        self.gaze_latent_encoder = MLPNetwork(num_layers=config.num_layer_mlp_enc,
                                              num_in=config.gaze_input_dim,
                                              num_hidden=config.gaze_input_dim,
                                              num_out=config.z_dim_gaze,
                                              non_linear=nn.LeakyReLU,
                                              non_linear_last=None).to(device)

        self.source_discrim = Discriminator().to(device)
        self.target_discrim = Discriminator().to(device)

        # Latent domain discriminator D_F
        self.latent_discriminator = MLPNetwork(num_layers=config.disc_num_layer,
                                          num_in=cat_latent_dim,
                                          num_hidden=cat_latent_dim,
                                          num_out=1,
                                          non_linear=nn.LeakyReLU,
                                          non_linear_last=None).to(device)

        # Task network T
        self.task_net = GazeHeadResNet(norm_layer='instance').to(device)

        # task network is fixed
        for param in self.task_net.parameters():
            param.requires_grad = False

        self.task_net = torch.nn.DataParallel(self.task_net)
        self.task_net.load_state_dict(torch.load(config.gazenet_savepath))
        self.task_net.eval()

    def push_modules_to_multi_gpu(self):
        self.image_encoder = torch.nn.DataParallel(self.image_encoder)
        self.generator = torch.nn.DataParallel(self.generator)
        self.gaze_latent_encoder = torch.nn.DataParallel(self.gaze_latent_encoder)
        self.source_discrim = torch.nn.DataParallel(self.source_discrim)
        self.target_discrim = torch.nn.DataParallel(self.target_discrim)
        self.latent_discriminator = torch.nn.DataParallel(self.latent_discriminator)

    def eval_mode_on(self):
        self.image_encoder.eval()
        self.generator.eval()
        self.gaze_latent_encoder.eval()
        self.source_discrim.eval()
        self.target_discrim.eval()
        self.latent_discriminator.eval()
        self.task_net.eval()

    def train_mode_on(self):
        self.image_encoder.train()
        self.generator.train()
        self.gaze_latent_encoder.train()
        self.source_discrim.train()
        self.target_discrim.train()
        self.latent_discriminator.train()
        self.task_net.eval()

    def save_model(self, current_step):
        models = {
            'image_encoder': self.image_encoder.state_dict(),
            'generator': self.generator.state_dict(),
            'target_discriminator': self.target_discrim.state_dict(),
            'source_discriminator': self.source_discrim.state_dict(),
            'latent_discriminator': self.latent_discriminator.state_dict(),
            'gaze_encoder': self.gaze_latent_encoder.state_dict(),
            'task_net': self.task_net.state_dict(),
        }

        p = os.path.join(self.config.save_path, "checkpoints")
        path = os.path.join(p, str(current_step) + '.pt')
        if not os.path.exists(p):
            os.makedirs(p)
        torch.save(models, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.image_encoder.load_state_dict(checkpoint['image_encoder'])
        self.generator.load_state_dict(checkpoint['generator'])
        self.target_discrim.load_state_dict(checkpoint['target_discriminator'])
        self.source_discrim.load_state_dict(checkpoint['source_discriminator'])
        self.latent_discriminator.load_state_dict(checkpoint['latent_discriminator'])
        self.gaze_latent_encoder.load_state_dict(checkpoint['gaze_encoder'])
        self.task_net.load_state_dict(checkpoint['task_net'])





