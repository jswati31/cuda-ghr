import torch
from torch import nn
import numpy as np
from torch.nn import functional as F


def rot_matrix_x(theta):
    """
    theta: measured in radians
    """
    mat = torch.zeros((3,3), dtype=torch.float32)
    mat[0, 0] = 1.
    mat[1, 1] = torch.cos(theta)
    mat[1, 2] = -torch.sin(theta)
    mat[2, 1] = torch.sin(theta)
    mat[2, 2] = torch.cos(theta)
    return mat

def rot_matrix_y(theta):
    """
    theta: measured in radians
    """
    mat = torch.zeros((3,3), dtype=torch.float32)
    mat[0, 0] = torch.cos(theta)
    mat[0, 2] = torch.sin(theta)
    mat[1, 1] = 1.
    mat[2, 0] = -torch.sin(theta)
    mat[2, 2] = torch.cos(theta)
    return mat

def rot_matrix_z(theta):
    """
    theta: measured in radians
    """
    mat = torch.zeros((3,3), dtype=torch.float32)
    mat[0, 0] = torch.cos(theta)
    mat[0, 1] = -torch.sin(theta)
    mat[1, 0] = torch.sin(theta)
    mat[1, 1] = torch.cos(theta)
    mat[2, 2] = 1.
    return mat


def pad_rotmat(theta):
    """theta = (3x3) rotation matrix"""
    return torch.hstack((theta, torch.zeros((3, 1))))

def pad_2d_rotmat(theta):
    """theta = (3x3) rotation matrix"""
    return torch.hstack((theta, torch.zeros((2, 1))))


def get_theta(angles):
    '''Construct a rotation matrix from angles.
    This uses the Euler angle representation. But
    it should also work if you use an axis-angle
    representation.
    '''

    bs = angles.shape[0]
    theta = torch.zeros((bs, 3, 4))

    angles_x = angles[:, 0]
    angles_y = angles[:, 1]
    angles_z = angles[:, 2]
    for i in range(bs):
        theta[i] = pad_rotmat(
            torch.mm(torch.mm(rot_matrix_z(angles_z[i]), rot_matrix_y(angles_y[i])),
                   rot_matrix_x(angles_x[i]))
        )

    return theta


def get_2d_theta(angles):
    '''Construct a rotation matrix from angles.
    This uses the Euler angle representation. But
    it should also work if you use an axis-angle
    representation.
    '''

    bs = angles.shape[0]
    theta = torch.zeros((bs, 3, 4))

    angles_x = angles[:, 0]
    angles_y = angles[:, 1]
    for i in range(bs):
        theta[i] = pad_rotmat(torch.mm(rot_matrix_y(angles_y[i]), rot_matrix_x(angles_x[i])))

    return theta


class Conv3dAdaIn(nn.Module):
    def __init__(self,
                 in_ch,
                 num_feature_maps,
                 z_size,
                 kernel_size=3,
                 double_conv=True,
                 mlp_non_linear=nn.LeakyReLU,
                 conv_non_linear=nn.LeakyReLU):
        super(Conv3dAdaIn, self).__init__()

        if double_conv:
            self.map_3d = nn.Sequential(
                nn.Conv3d(in_ch, num_feature_maps, kernel_size, padding="same"),
                conv_non_linear(),
                nn.Conv3d(num_feature_maps, num_feature_maps, kernel_size, padding="same"),
            )
        else:
            self.map_3d = nn.Sequential(
                nn.Conv3d(in_ch, num_feature_maps, kernel_size, padding="same"),
            )

        self.adain = AdaIn(out_ch=num_feature_maps, z_size=z_size)

        self.nl = conv_non_linear()

    def forward(self, inputs):
        x = inputs['x']
        z = inputs['z']

        x = self.map_3d(x)
        x = self.nl(x)
        x = self.adain(x, z)
        return x


# Block of 2d Convolution(s), followed by instance norm
class Conv2dAdaIn(nn.Module):
    def __init__(self,
                 in_ch,
                 num_feature_maps,
                 z_size,
                 double_conv=False,
                 kernel_size=3,
                 mlp_non_linear=nn.LeakyReLU,
                 conv_non_linear=nn.LeakyReLU):
        super(Conv2dAdaIn, self).__init__()

        if double_conv:
            self.map_2d = nn.Sequential(
                nn.Conv2d(in_ch, num_feature_maps, kernel_size, padding="same"),
                conv_non_linear(),
                nn.Conv2d(num_feature_maps, num_feature_maps, kernel_size, padding="same"),
            )
        else:
            self.map_2d = nn.Sequential(
                nn.Conv2d(in_ch, num_feature_maps, kernel_size, padding="same"),
            )

        self.adain = AdaIn(out_ch=num_feature_maps, z_size=z_size)

        self.nl = conv_non_linear()

    def forward(self, inputs):
        x = inputs['x']
        z = inputs['z']

        x = self.map_2d(x)
        x = self.nl(x)
        x = self.adain(x, z)
        return x


# Adaptive Instance Norm
class AdaIn(nn.Module):
    def __init__(self,
                 out_ch,
                 z_size

                 ):
        super(AdaIn, self).__init__()

        # construct the layers feeding into the AdaIn
        self.adain_mlp = nn.Linear(z_size, out_ch * 2)  # both var and mean

        self.instance_norm_3d = nn.InstanceNorm3d(out_ch, affine=True)
        self.instance_norm_2d = nn.InstanceNorm2d(out_ch, affine=True)

    def _rshp2d(self, z):
        return z.view(-1, z.size(1), 1, 1)

    def _rshp3d(self, z):
        return z.view(-1, z.size(1), 1, 1, 1)

    def _split(self, z):
        len_ = z.size(1)
        mean = z[:, 0:(len_ // 2)]
        var = F.softplus(z[:, (len_ // 2):])
        return mean, var

    def forward(self, x, z):
        # get adaptive instance norm parameters
        if len(x.shape) == 5:  # 3d input
            z_mean, z_var = self._split(self._rshp3d(self.adain_mlp(z)))
            x = self.instance_norm_3d(x)
        else:  # 2d input
            z_mean, z_var = self._split(self._rshp2d(self.adain_mlp(z)))
            x = self.instance_norm_2d(x)

        # now apply instance norm
        x = x * z_var + z_mean

        return x


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class HologanGenerator(nn.Module):
    def __init__(self, latent_dim, output_shape, gen_output_activation=nn.Tanh):
        super(HologanGenerator, self).__init__()

        self.ups_3d = nn.Upsample(scale_factor=2, mode='nearest')
        self.ups_2d = nn.Upsample(scale_factor=2, mode='nearest')

        self.output_img_shape = output_shape
        if self.output_img_shape[0] == self.output_img_shape[1]:
            self.const_shape = (512, 4, 4, 4)
        else:
            self.const_shape = (512, 4, 2, 8)

        n_features_in_first_layer = 256

        nl_f = nn.LeakyReLU

        self.zero_input = LambdaLayer(lambda x: self.get_zero_inputs(x))

        self.learned_input_layer = nn.Linear(1, np.prod(self.const_shape))

        # pre-rotation function
        self.map_3d_0 = Conv3dAdaIn(
            in_ch=n_features_in_first_layer * 2,
            num_feature_maps=n_features_in_first_layer,
            kernel_size=3,
            double_conv=False,
            z_size=latent_dim,
            conv_non_linear=nl_f)

        self.map_3d_1 = Conv3dAdaIn(
            in_ch=n_features_in_first_layer,
            num_feature_maps=n_features_in_first_layer // 2,
            kernel_size=3,
            double_conv=False,
            z_size=latent_dim,
            conv_non_linear=nl_f)

        # NOTE: rotation (done in call)

        # post-rotation function
        self.map_3d_post = nn.Sequential(
            nn.Conv3d(n_features_in_first_layer // 2, n_features_in_first_layer // 4, 3, padding="same"),
            nl_f(),
            nn.Conv3d(n_features_in_first_layer // 4, n_features_in_first_layer // 4, 3, padding="same"),
            nl_f()
        )

        pnf = (2 * n_features_in_first_layer // 8) * (4 ** 2)  # 512

        self.projection_conv = nn.Sequential(
            nn.Conv2d(pnf, 512, kernel_size=1, padding="same"),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU()
        )

        # 2d mapping
        self.map_2d_0 = Conv2dAdaIn(
            in_ch=2 * n_features_in_first_layer,
            num_feature_maps=n_features_in_first_layer,
            kernel_size=4,
            double_conv=False,
            z_size=latent_dim,
            conv_non_linear=nl_f)

        self.map_2d_1 = Conv2dAdaIn(
            in_ch=n_features_in_first_layer,
            num_feature_maps=n_features_in_first_layer // 4, kernel_size=4, double_conv=False,
            z_size=latent_dim,
            conv_non_linear=nl_f)

        self.map_2d_2 = Conv2dAdaIn(
            in_ch=n_features_in_first_layer // 4,
            num_feature_maps=n_features_in_first_layer // 8, kernel_size=4, double_conv=False,
            z_size=latent_dim,
            conv_non_linear=nl_f)

        last_layer_inp_dim = n_features_in_first_layer // 8

        if self.output_img_shape[0] > 128:
            self.map_2d_2b = Conv2dAdaIn(
                in_ch=n_features_in_first_layer // 8,

                num_feature_maps=n_features_in_first_layer // 8, kernel_size=4, double_conv=False,
                z_size=latent_dim,
                conv_non_linear=nl_f)
            last_layer_inp_dim = n_features_in_first_layer // 8

        if self.output_img_shape[0] > 256:
            self.map_2d_2c = Conv2dAdaIn(
                in_ch=n_features_in_first_layer // 8,

                num_feature_maps=n_features_in_first_layer // 16, kernel_size=4, double_conv=False,
                z_size=latent_dim,
                conv_non_linear=nl_f)
            last_layer_inp_dim = n_features_in_first_layer // 16

        self.map_final = nn.Conv2d(last_layer_inp_dim, 3, 4, padding="same")
        self.final_act = gen_output_activation()

    def get_zero_inputs(self, input_layer):
        zero_input = torch.zeros((1, 1)).to(input_layer.device)
        zero_input = torch.tile(zero_input, (input_layer.shape[0], 1))

        return zero_input

    def build_input_dict(self, latent_vector, rotation):
        input_dict = {}
        if isinstance(latent_vector, list):
            input_dict["z_3d_0"] = latent_vector[0]
            input_dict["z_3d_1"] = latent_vector[1]

            input_dict["z_2d_0"] = latent_vector[2]
            input_dict["z_2d_1"] = latent_vector[3]
            input_dict["z_2d_2"] = latent_vector[4]
        else:
            input_dict["z_3d_0"] = latent_vector
            input_dict["z_3d_1"] = latent_vector

            input_dict["z_2d_0"] = latent_vector
            input_dict["z_2d_1"] = latent_vector
            input_dict["z_2d_2"] = latent_vector
        input_dict["rotation"] = rotation

        return input_dict

    def stn(self, x, theta):
        # theta must be (Bs, 3, 4) = [R|t]
        # theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        out = F.grid_sample(x, grid, padding_mode='zeros')
        return out

    def forward(self, inputs):
        if not isinstance(inputs, dict):
            inputs = self.build_input_dict(inputs[0], inputs[1])

        zeros = self.zero_input(inputs['z_3d_0'])

        x = self.learned_input_layer(zeros)
        x = x.reshape(-1, *(self.const_shape))

        # upsample by factor of 2
        x = self.ups_3d(x)

        # transform
        x = self.map_3d_0({'x': x, 'z': inputs['z_3d_0']})

        x = self.ups_3d(x)

        x = self.map_3d_1({'x': x, 'z': inputs['z_3d_1']})

        theta = get_theta(inputs["rotation"]).to(inputs["rotation"].device)

        x = self.stn(x, theta)
        # 'rendering' layers
        x = self.map_3d_post(x)

        # ...including the reshape
        x_s = list(x.shape)
        if x_s[0] is None:
            x_s[0] = -1
        x = x.view(-1, x.size(1) * x.size(2), x.size(3), x.size(4))

        x = self.projection_conv(x)

        x = self.map_2d_0({'x': x, 'z': inputs['z_2d_0']})
        x = self.ups_2d(x)

        x = self.map_2d_1({'x': x, 'z': inputs['z_2d_1']})
        x = self.ups_2d(x)

        x = self.map_2d_2({'x': x, 'z': inputs['z_2d_2']})
        x = self.ups_2d(x)

        if self.output_img_shape[0] > 128:
            x = self.map_2d_2b({'x': x, 'z': inputs['z_2d_2']})
            x = self.ups_2d(x)
        if self.output_img_shape[0] > 256:
            x = self.map_2d_2c({'x': x, 'z': inputs['z_2d_2']})
            x = self.ups_2d(x)

        x = self.final_act(self.map_final(x))

        return x, None

