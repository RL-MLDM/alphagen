from typing import Callable, Tuple, Union

import torch as th
from torch import nn

from stable_baselines3.common.utils import get_device


class RNDNet(nn.Module):
    def __init__(
        self,
        module_factory: Callable[[], nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        self.device = device
        self.target = module_factory().to(device).requires_grad_(False)
        self.predictor = module_factory().to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        predicted = self.predictor(features)
        target = self.target(features)
        return predicted, target
