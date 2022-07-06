from typing import Tuple

import torch
from torch import nn, Tensor
from torch.distributions import Categorical, Normal

from alphagen.models.model import TokenEmbedding, PositionalEncoding
from alphagen.models.tokens import *
from alphagen.data.expression import *


class Policy:
    def __init__(
        self,
        n_encoder_layers: int,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float,
        operators: List[Type[Operator]],
        delta_time_range: Tuple[int, int],
        device: torch.device
    ):
        super().__init__()
        self.encoder = Encoder(
            n_encoder_layers=n_encoder_layers,
            d_model=d_model,
            n_head=n_head,
            d_ffn=d_ffn,
            dropout=dropout,
            delta_time_range=delta_time_range,
            device=device
        )
        self.decoder = Decoder(
            d_model=d_model,
            operators=operators,
            delta_time_range=delta_time_range,
            device=device
        )

    def get_action(self, state: List[Token], info: dict) -> Token:
        encoding = self.encoder(state).mean(dim=0).reshape(-1)
        action, log_prob = self.decoder(encoding, info)
        return action

    def encode(self, state: List[Token]):
        return self.encoder(state)


class Encoder(nn.Module):
    def __init__(
        self,
        n_encoder_layers: int,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float,
        delta_time_range: Tuple[int, int],
        device: torch.device
    ):
        super().__init__()
        self._device = device
        self._d_model = d_model
        self._dt_range = delta_time_range

        self._token_emb = TokenEmbedding(d_model, [], delta_time_range, device)
        self._pos_enc = PositionalEncoding(d_model).to(device)
        self._encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head, dim_feedforward=d_ffn,
                dropout=dropout, device=device
            ),
            n_encoder_layers,
            norm=nn.LayerNorm(d_model, device=device)
        )

    def forward(self, tokens: List[Token]) -> Tensor:
        res = self._token_emb(tokens)
        res = self._pos_enc(res)
        res = self._encoder.forward(res.unsqueeze(1))
        return res.squeeze(1)


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        operators: List[Type[Operator]],
        delta_time_range: Tuple[int, int],
        device: torch.device
    ):
        super().__init__()
        self._operators = operators
        self._dt_range = delta_time_range

        self._decision_linear = nn.Linear(d_model, 5, device=device)

        self._op_linear = nn.Linear(d_model, len(operators), device=device)
        self._feat_linear = nn.Linear(d_model, len(FeatureType), device=device)
        self._const_linear = nn.Linear(d_model, 2, device=device)
        self._dt_linear = nn.Linear(
            d_model,
            delta_time_range[1] - delta_time_range[0],
            device=device
        )

    def forward(self, res, info: dict) -> Tuple[Token, float]:
        logits = self._decision_linear(res)
        for i in range(5):
            if not info['select'][i]:
                logits[i] = -1e10
        select_dist = Categorical(logits=logits)
        idx = select_dist.sample()
        select_log_prob = select_dist.log_prob(idx)
        idx = int(idx)
        if idx == 0:    # Operators
            logits = self._op_linear(res)
            for i, op in enumerate(self._operators):
                if not info['op'][op.category_type()]:
                    logits[i] = -1e10
            dist = Categorical(logits=logits)
            idx = dist.sample()
            log_prob = select_log_prob + dist.log_prob(idx)
            return OperatorToken(self._operators[int(idx)]), log_prob
        elif idx == 1:  # Features
            dist = Categorical(logits=self._feat_linear(res))
            idx = dist.sample()
            log_prob = select_log_prob + dist.log_prob(idx)
            return FeatureToken(FeatureType(int(idx))), log_prob
        elif idx == 2:  # Constants
            affine: Tensor = self._const_linear(res)
            mu, sigma = affine[0], affine[1].exp()
            dist = Normal(mu, sigma)
            z = dist.sample()
            log_prob = select_log_prob + dist.log_prob(z)
            return ConstantToken(float(z)), log_prob
        elif idx == 3:  # Date Delta
            dist = Categorical(logits=self._dt_linear(res))
            idx = dist.sample()
            log_prob = dist.log_prob(idx)
            dt = self._dt_range[0] + int(idx)
            return DeltaTimeToken(dt), log_prob
        else:           # End
            return SequenceIndicatorToken(SequenceIndicatorType.SEP), select_log_prob


if __name__ == '__main__':
    policy = Policy(
        n_encoder_layers=6,
        d_model=512,
        n_head=8,
        d_ffn=2048,
        dropout=0.1,
        operators=[Add, Sub, Ref, Abs],
        delta_time_range=(1, 31),
        device=torch.device("cuda:0")
    )

    import qlib
    from qlib.constant import REG_CN
    from alphagen.rl.env import AlphaEnvCore

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    env = AlphaEnvCore(
        instrument='csi300',
        start_time='2016-01-01',
        end_time='2018-12-31'
    )

    for i in range(50):
        state, info = env.reset()
        while True:
            action = policy.get_action(state, info)
            state, reward, done, info = env.step(action)
            # print(f'next_state: {state}, reward: {reward}, done: {done}')
            if done:
                seq = str(env._builder.get_tree()) if env._builder.is_valid() else 'Invalid'
                print(f'seq: {seq}, reward: {reward}')
                break
