import torch.nn as nn


def build_layered_module(input_dim, net_arch=[256, 256, 256], activation_fn=nn.ReLU, output_fn=None):
    return nn.Sequential(*(
        [nn.Linear(input_dim, net_arch[0])] +
        sum([[activation_fn, nn.Linear(_from, _to)] for _from, _to in zip(net_arch, net_arch[1:] + [1])], []) +
        ([output_fn] if output_fn else [])
    ))
