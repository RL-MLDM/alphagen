from typing import Dict, Type, Union, List, cast

import torch as th
from torch import nn
from stable_baselines3.common.utils import get_device


class MlpExtractor(nn.Module):
    """
    MLP feature extractor adapted from Stable Baselines.

    [55, {"vf": [255, 255], "pi": [128]} -> shared net: [55], vf net: [255, 255], pi net layer: [128]
    [128, 128] -> shared net: [128, 128]

    :param feature_dim: Dimension of the feature vector
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ):
        super(MlpExtractor, self).__init__()
        device = get_device(device)
        self.device = device
        self.nets = nn.ModuleDict()

        has_separate = len(net_arch) > 0 and isinstance(net_arch[-1], dict)
        self.separate_arch: Dict[str, List[int]]
        if has_separate:
            net_arch = net_arch.copy()
            self.separate_arch = cast(Dict[str, List[int]], net_arch.pop())
        else:
            self.separate_arch = {}
        self.shared_arch = cast(List[int], net_arch)

        def construct_seq(sizes: List[int]) -> nn.Sequential:
            layers = []
            for i, o in zip(sizes[:-1], sizes[1:]):
                layers.append(nn.Linear(i, o))
                layers.append(activation_fn())
            return nn.Sequential(*layers).to(device)

        shared_arch = [feature_dim, *self.shared_arch]
        self.shared_latent_dim = shared_arch[-1]
        self.shared_net = construct_seq(shared_arch)
        for name, arch in self.separate_arch.items():
            self.nets[name] = construct_seq([shared_arch[-1], *arch])

    def forward(self, features: th.Tensor) -> Dict[str, th.Tensor]:
        shared_latent = self.shared_net(features)
        return {name: net(shared_latent) for name, net in self.nets.items()}

    def forward_only(self, features: th.Tensor, names: List[str]) -> List[th.Tensor]:
        shared_latent = self.shared_net(features)
        return [self._forward_sep(shared_latent, name) for name in names]

    def _forward_sep(self, shared_latent: th.Tensor, name: str) -> th.Tensor:
        return self.nets[name](shared_latent) if name in self.nets else shared_latent

    def latent_dim(self, name: str) -> int:
        def last_or_shared_dim(lst: List[int]):
            return self.shared_latent_dim if len(lst) == 0 else lst[-1]

        return last_or_shared_dim(self.separate_arch.get(name, []))
