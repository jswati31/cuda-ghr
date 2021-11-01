import torch.nn as nn
import torch


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def replace_instance(module, name):
    '''
    Recursively put desired instance norm in nn.module module.

    set module = net to start code.
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and put instance norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d:
            new_bn = torch.nn.InstanceNorm2d(target_attr.num_features, target_attr.eps, target_attr.momentum, target_attr.affine,
                                          track_running_stats=False)
            setattr(module, attr_str, new_bn)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_instance(immediate_child_module, name)


def replace_bn(module, name):
    '''
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d:
            new_bn = torch.nn.BatchNorm2d(target_attr.num_features, target_attr.eps, target_attr.momentum, target_attr.affine,
                                          track_running_stats=False)
            setattr(module, attr_str, new_bn)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_bn(immediate_child_module, name)


def replace_relu(module, name):
    '''
    Recursively put desired LeakyRelu in nn.module module.
    set module = net to start code.
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and put LeakyRelu if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == torch.nn.ReLU:
            new_act = torch.nn.LeakyReLU(target_attr.inplace)
            setattr(module, attr_str, new_act)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_relu(immediate_child_module, name)

