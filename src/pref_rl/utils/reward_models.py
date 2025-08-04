import torch
import torch.nn as nn

from .model import build_layered_module


class RewardModel(nn.Module):
    def __init__(self, input_dim, ensemble_size=3, **kwargs):
        super().__init__()
        self.members = nn.ModuleList([self._build_member(input_dim, **kwargs) for _ in range(ensemble_size)])


    def _build_member(self, input_dim, net_arch=[256, 256, 256], activation_fn=nn.LeakyReLU(), output_fn=nn.Tanh()):
        return build_layered_module(input_dim, net_arch=net_arch, activation_fn=activation_fn, output_fn=output_fn)


    def forward(self, x):
        return torch.stack([member(x) for member in self.members])


class MultiHeadMember(nn.Module):
    def __init__(self, trunk, pref_head, aux_head, output_fn):
        super().__init__()
        self.trunk = trunk
        self.pref_head = pref_head
        self.aux_head = aux_head
        self.output_fn = output_fn


    def forward(self, x):
        trunk_output = self.trunk(x)
        pref_output = self.pref_head(trunk_output)
        return self.output_fn(pref_output)


    def auxiliary(self, x):
        trunk_output = self.trunk(x)
        return self.aux_head(trunk_output)


    def freeze_pref(self):
        for param in self.trunk.parameters():
            param.requires_grad = False
        for param in self.pref_head.parameters():
            param.requires_grad = False


    def unfreeze_pref(self):
        for param in self.trunk.parameters():
            param.requires_grad = True
        for param in self.pref_head.parameters():
            param.requires_grad = True


class MultiHeadRewardModel(nn.Module):
    def __init__(self, input_dim, ensemble_size=3, **kwargs):
        super().__init__()
        self.members = nn.ModuleList([self._build_member(input_dim, **kwargs) for _ in range(ensemble_size)])


    def _build_member(self, input_dim, n_heads=2, net_arch=[256, 256, 256], activation_fn=nn.LeakyReLU(), output_fn=nn.Tanh()):
        trunk = build_layered_module(
            input_dim,
            net_arch=net_arch[:-1],
            activation_fn=activation_fn,
            output_fn=None,
            output_dim=net_arch[-1],
        )
        pref_head = nn.Linear(net_arch[-1], 1)
        aux_head = nn.Linear(net_arch[-1], n_heads - 1)
        
        return MultiHeadMember(trunk, pref_head, aux_head, output_fn)


    def forward(self, x):
        return torch.stack([member(x) for member in self.members])
