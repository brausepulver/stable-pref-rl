import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy

from ..utils.model import build_layered_module


class SharedMlpActorCriticPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        assert isinstance(self.net_arch, list), "net_arch must be a list of integers"

        self.shared_net = build_layered_module(
            input_dim=self.features_dim,
            output_dim=self.net_arch[-1],
            net_arch=self.net_arch[:-1],
            activation_fn=nn.Tanh(),
            output_fn=None
        )
        latent_dim_pi = self.net_arch[-1]
        latent_dim_vf = self.net_arch[-1]

        class CustomMlpExtractor(nn.Module):
            def __init__(self, shared_net):
                super().__init__()
                self.shared_net = shared_net
                self.latent_dim_pi = latent_dim_pi
                self.latent_dim_vf = latent_dim_vf

            def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
                return self.forward_actor(features), self.forward_critic(features)

            def forward_actor(self, features: th.Tensor) -> th.Tensor:
                return self.shared_net(features)

            def forward_critic(self, features: th.Tensor) -> th.Tensor:
                return self.shared_net(features)

        self.mlp_extractor = CustomMlpExtractor(self.shared_net)
