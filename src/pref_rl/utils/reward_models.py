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
    def __init__(self, trunk, head, output_fn):
        super().__init__()
        self.trunk = trunk
        self.head = head
        self.output_fn = output_fn


    def forward(self, x):
        output = self.head(self.trunk(x))
        first_head = self.output_fn(output[..., 0:1])
        return first_head


    def auxiliary(self, x):
        output = self.head(self.trunk(x))
        return output[..., 1:]


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
        head = nn.Linear(net_arch[-1], n_heads)
        
        return MultiHeadMember(trunk, head, output_fn)


    def forward(self, x):
        return torch.stack([member(x) for member in self.members])
