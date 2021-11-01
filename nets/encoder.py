
import torch
import torch.nn as nn
from .densenet import (
    DenseNetInitialLayers,
    DenseNetBlock,
    DenseNetTransitionDown,
)


class Encoder(nn.Module):

    def __init__(self, z_dim_app, num_blocks=4, growth_rate=32, bias=True, activation_fn=nn.LeakyReLU,
                 normalization_fn=nn.InstanceNorm2d
                 ):
        super(Encoder, self).__init__()

        # The meaty parts
        self.encoder_ = DenseNetEncoder(
            num_blocks=num_blocks,
            growth_rate=growth_rate,
            activation_fn=activation_fn,
            normalization_fn=normalization_fn,
        )
        c_now = list(self.children())[-1].c_now

        self.fc_appearance = self.linear(c_now, z_dim_app, bias)

    def linear(self, f_in, f_out, bias):
        fc = nn.Linear(f_in, f_out, bias=bias)
        nn.init.kaiming_normal_(fc.weight.data)
        if bias:
            nn.init.constant_(fc.bias.data, val=0)
        return fc

    def forward(self, data1):
        data1 = self.encoder_(data1)
        data1 = data1.mean(-1).mean(-1)  # Global-Average Pooling
        z_app_enc = self.fc_appearance(data1)
        return z_app_enc


class DenseNetEncoder(nn.Module):

    def __init__(self, growth_rate=8, num_blocks=4, num_layers_per_block=4,
                 p_dropout=0.0, compression_factor=1.0,
                 activation_fn=nn.ReLU, normalization_fn=nn.InstanceNorm2d):
        super(DenseNetEncoder, self).__init__()
        self.c_at_end_of_each_scale = []

        # Initial down-sampling conv layers
        self.initial = DenseNetInitialLayers(growth_rate=growth_rate,
                                             activation_fn=activation_fn,
                                             normalization_fn=normalization_fn)
        c_now = list(self.children())[-1].c_now
        self.c_at_end_of_each_scale += list(self.children())[-1].c_list

        assert (num_layers_per_block % 2) == 0
        for i in range(num_blocks):
            i_ = i + 1
            # Define dense block
            self.add_module('block%d' % i_, DenseNetBlock(
                c_now,
                num_layers=num_layers_per_block,
                growth_rate=growth_rate,
                p_dropout=p_dropout,
                activation_fn=activation_fn,
                normalization_fn=normalization_fn,
            ))
            c_now = list(self.children())[-1].c_now
            self.c_at_end_of_each_scale.append(c_now)

            # Define transition block if not last layer
            if i < (num_blocks - 1):
                self.add_module('trans%d' % i_, DenseNetTransitionDown(
                    c_now, p_dropout=p_dropout,
                    compression_factor=compression_factor,
                    activation_fn=activation_fn,
                    normalization_fn=normalization_fn,
                ))
                c_now = list(self.children())[-1].c_now
            self.c_now = c_now

    def forward(self, x):
        # Apply initial layers and dense blocks
        for name, module in self.named_children():
            if name == 'initial':
                x, prev_scale_x = module(x)
            else:
                x = module(x)
        return x


class MLPNetwork(nn.Module):
    def __init__(self, num_layers, num_in, num_hidden, num_out, non_linear=nn.LeakyReLU, non_linear_last=None):
        super(MLPNetwork, self).__init__()

        self.map = []
        self.num_in = num_in
        self.num_out = num_out

        current_num_in = num_in
        # hidden layers
        for _ in range(num_layers - 1):
            self.map.append(nn.Linear(current_num_in, num_hidden))
            self.map.append(non_linear())
            current_num_in = num_hidden

        # last layer
        self.map.append(nn.Linear(current_num_in, num_out))
        if non_linear_last is not None:
            self.map.append(non_linear_last())

        self.map = nn.Sequential(*self.map)

    def forward(self, inputs):
        return self.map(inputs)





