from typing import Tuple, Optional

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
        self.device = device
        self.decoder = Decoder(
            n_encoder_layers=n_encoder_layers,
            d_model=d_model,
            n_head=n_head,
            d_ffn=d_ffn,
            dropout=dropout,
            delta_time_range=delta_time_range,
            device=device
        )
        self.policy_net = PolicyNet(
            d_model=d_model,
            operators=operators,
            delta_time_range=delta_time_range,
            device=device
        )
        self.value_net = ValueNet(d_model=d_model, device=device)

    def _decode_flatten(self, state: List[Token]) -> Tensor:
        return self.decoder(state).mean(dim=0).reshape(-1)

    def get_action(self, state: List[Token], info: dict,
                   action: Optional[Token] = None) -> Tuple[Token, Tensor]:
        return self.policy_net(self._decode_flatten(state), info, action)

    def get_value(self, state: List[Token], info: dict) -> Tensor:
        return self.value_net(self._decode_flatten(state), info)

    def get_action_value(self, state: List[Token], info: dict,
                         action: Optional[Token] = None) -> Tuple[Token, Tensor, Tensor]:
        repr = self._decode_flatten(state)
        act_logp: Tuple[Token, Tensor] = self.policy_net(repr, info, action)
        value = self.value_net(repr, info)
        return *act_logp, value

    def decode(self, state: List[Token]):
        return self.decoder(state)

    def policy_parameters(self):
        yield from self.decoder.parameters()
        yield from self.policy_net.parameters()

    def value_parameters(self):
        yield from self.decoder.parameters()
        yield from self.value_net.parameters()

    def parameters(self):
        yield from self.decoder.parameters()
        yield from self.policy_net.parameters()
        yield from self.value_net.parameters()

    def save(self, path: str) -> None:
        torch.save({
            "decoder": self.decoder.state_dict(),
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict()
        }, path)

    def load(self, path: str) -> None:
        dicts = torch.load(path)
        self.decoder.load_state_dict(dicts["decoder"])
        self.policy_net.load_state_dict(dicts["policy_net"])
        self.value_net.load_state_dict(dicts["value_net"])


class Decoder(nn.Module):
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


class PolicyNet(nn.Module):
    def __init__(
        self,
        d_model: int,
        operators: List[Type[Operator]],
        delta_time_range: Tuple[int, int],
        device: torch.device
    ):
        super().__init__()
        self._device = device

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

    def _tensor(self, value) -> Tensor: return torch.tensor(value, device=self._device)

    def _sample_token_category(self, dist: Categorical, action: Optional[Token]) -> Tuple[int, Tensor]:
        if action is None:
            idx = dist.sample()
        elif isinstance(action, OperatorToken):
            idx = self._tensor(0)
        elif isinstance(action, FeatureToken):
            idx = self._tensor(1)
        elif isinstance(action, ConstantToken):
            idx = self._tensor(2)
        elif isinstance(action, DeltaTimeToken):
            idx = self._tensor(3)
        elif isinstance(action, SequenceIndicatorToken):
            idx = self._tensor(4)
        else:
            assert False

        return int(idx), dist.log_prob(idx)

    def _sample_operator(self, dist: Categorical, action: Optional[Token]) -> Tuple[Token, Tensor]:
        if action is None:
            idx = dist.sample()
            action = OperatorToken(self._operators[int(idx)])
        else:
            assert isinstance(action, OperatorToken)
            idx = self._tensor(self._operators.index(action.operator))
        return action, dist.log_prob(idx)

    def _sample_feature(self, dist: Categorical, action: Optional[Token]) -> Tuple[Token, Tensor]:
        if action is None:
            idx = dist.sample()
            action = FeatureToken(FeatureType(int(idx)))
        else:
            assert isinstance(action, FeatureToken)
            idx = self._tensor(int(action.feature))
        return action, dist.log_prob(idx)

    def _sample_constant(self, dist: Normal, action: Optional[Token]) -> Tuple[Token, Tensor]:
        if action is None:
            z = dist.sample()
            action = ConstantToken(float(z))
        else:
            assert isinstance(action, ConstantToken)
            z = self._tensor(action.constant)
        return action, dist.log_prob(z)

    def _sample_dt(self, dist: Categorical, action: Optional[Token]) -> Tuple[Token, Tensor]:
        if action is None:
            idx = dist.sample()
            action = DeltaTimeToken(int(idx) + self._dt_range[0])
        else:
            assert isinstance(action, DeltaTimeToken)
            idx = self._tensor(action.delta_time - self._dt_range[0])
        return action, dist.log_prob(idx)

    def forward(self, res: Tensor, info: dict,
                action: Optional[Token] = None) -> Tuple[Token, Tensor]:
        logits = self._decision_linear(res)
        for i in range(5):
            if not info["select"][i]:
                logits[i] = -1e10
        select_dist = Categorical(logits=logits)
        idx, select_log_prob = self._sample_token_category(select_dist, action)

        if idx == 0:    # Operators
            logits = self._op_linear(res)
            for i, op in enumerate(self._operators):
                if not info["op"][op.category_type()]:
                    logits[i] = -1e10
            dist = Categorical(logits=logits)
            token, logp = self._sample_operator(dist, action)
            return token, select_log_prob + logp

        elif idx == 1:  # Features
            dist = Categorical(logits=self._feat_linear(res))
            token, logp = self._sample_feature(dist, action)
            return token, select_log_prob + logp

        elif idx == 2:  # Constants
            affine: Tensor = self._const_linear(res)
            mu, sigma = affine[0], affine[1].exp()
            dist = Normal(mu, sigma)
            token, logp = self._sample_constant(dist, action)
            return token, select_log_prob + logp

        elif idx == 3:  # Date Delta
            dist = Categorical(logits=self._dt_linear(res))
            token, logp = self._sample_dt(dist, action)
            return token, select_log_prob + logp

        else:           # End
            return SEP_TOKEN, select_log_prob


class ValueNet(nn.Module):
    def __init__(self, d_model: int, device: torch.device):
        super().__init__()
        self._value_linear = nn.Linear(d_model, 1, device=device)

    def forward(self, res: Tensor, info: dict) -> Tensor:
        value = torch.sigmoid(self._value_linear(res)).squeeze()
        return value


def main():
    policy = Policy(
        n_encoder_layers=6,
        d_model=512,
        n_head=8,
        d_ffn=2048,
        dropout=0.1,
        operators=[Add, Sub, Mul, Div, Ref, Abs, EMA, Sum, Mean, Std],
        delta_time_range=(1, 31),
        device=torch.device("cuda:0")
    )

    import qlib
    from qlib.constant import REG_CN
    from alphagen.rl.env import AlphaEnvCore

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    env = AlphaEnvCore(
        instrument="csi300",
        start_time="2016-01-01",
        end_time="2018-12-31"
    )

    for _ in range(50):
        state, info = env.reset()
        while True:
            action, log_prob = policy.get_action(state, info)
            state, reward, done, info = env.step(action)
            if done:
                seq = str(env._builder.get_tree()) if env._builder.is_valid() else "Invalid"
                print(f"seq: {seq}, log(p): {log_prob}, reward: {reward}")
                break


if __name__ == '__main__':
    main()
