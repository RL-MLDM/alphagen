import gym
import gym.spaces

from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from alphagen.models.model import PositionalEncoding
from alphagen.data.tokens import *
from alphagen.data.expression import *


class Decoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        n_layers: int,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float,
        device: torch.device
    ):
        super().__init__(observation_space, d_model)

        assert isinstance(observation_space, gym.spaces.Box)
        n_actions = observation_space.high[0]   # type: ignore

        self._device = device
        self._d_model = d_model
        self._n_actions: float = n_actions

        self._token_emb = nn.Embedding(n_actions + 1, d_model, 0)   # Last one is [BEG]
        self._pos_enc = PositionalEncoding(d_model).to(device)

        # Actually an encoder for now
        self._decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head, dim_feedforward=d_ffn,
                dropout=dropout, batch_first=True, device=device
            ),
            n_layers,
            norm=nn.LayerNorm(d_model, device=device)
        )

    def forward(self, obs: Tensor) -> Tensor:
        batch_size = obs.size(0)
        begins = torch.full(size=(batch_size, 1), fill_value=self._n_actions,
                            dtype=torch.long, device=obs.device)
        obs = torch.cat((begins, obs.type(torch.long)), dim=1)      # (bs, len)
        pad_mask = obs == 0
        res = self._token_emb(obs)                                  # (bs, len, d_model)
        res = self._pos_enc(res)                                    # (bs, len, d_model)
        res = self._decoder(res, src_key_padding_mask=pad_mask)     # (bs, len, d_model)
        return res.mean(dim=1)                                      # (bs, d_model)
