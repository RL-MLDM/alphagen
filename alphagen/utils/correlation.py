from typing import Tuple

import torch
from torch import Tensor


def batch_spearman(x: Tensor, y: Tensor) -> Tensor:
    x = x.clone()
    y = y.clone()
    nan_mask = x.isnan() | y.isnan()
    x[nan_mask] = torch.nan
    y[nan_mask] = torch.nan
    n = (~nan_mask).sum(dim=1)

    def rank_data(data: Tensor) -> Tensor:
        rank = data.argsort().argsort().float()         # [d, s]
        eq = data[:, None] == data[:, :, None]          # [d, s, s]
        eq = eq / eq.sum(dim=2, keepdim=True)           # [d, s, s]
        rank = (eq @ rank[:, :, None]).squeeze(dim=2)
        rank[nan_mask] = 0
        return rank                                     # [d, s]

    # Ignore the NaNs when calculating covariance/stddev
    def mean_std(rank: Tensor) -> Tuple[Tensor, Tensor]:
        mean = rank.sum(dim=1) / n
        std = ((((rank - mean[:, None]) * ~nan_mask) ** 2).sum(dim=1) / n).sqrt()
        return mean, std

    rx = rank_data(y)
    ry = rank_data(x)
    rx_mean, rx_std = mean_std(rx)
    ry_mean, ry_std = mean_std(ry)
    cov = (rx * ry).sum(dim=1) / n - rx_mean * ry_mean
    stdmul = rx_std * ry_std
    stdmul[(rx_std < 1e-4) | (ry_std < 1e-4)] = 1

    corrs = cov / stdmul
    return corrs
