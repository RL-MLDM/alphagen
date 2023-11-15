from typing import Tuple, List
import math

import torch
from torch import nn, Tensor
from torch.distributions import Categorical, Normal
from alphagen.data.expression import Operators
from alphagen.data.tokens import *


# Deprecated!
class TokenEmbedding(nn.Module):
    def __init__(self,
                 d_model: int,
                 operators: List[Type[Operator]],
                 delta_time_range: Tuple[int, int],
                 device: torch.device):
        super().__init__()
        self._d_model = d_model
        self._operators = operators
        self._delta_time_range = delta_time_range
        self._device = device

        self._const_linear = nn.Linear(1, d_model, device=device)
        dt_count = delta_time_range[1] - delta_time_range[0]
        total_emb = (len(SequenceIndicatorType) + len(FeatureType) +
                     len(Operators) + dt_count)
        self._emb = nn.Embedding(total_emb, d_model, device=device)

    def forward(self, tokens: List[Token]) -> Tensor:
        const_idx: List[int] = []
        consts: List[float] = []
        emb_idx: List[int] = []
        emb_type_idx: List[int] = []

        feat_offset = len(SequenceIndicatorType)
        op_offset = feat_offset + len(FeatureType)
        dt_offset = op_offset + len(Operators)

        for i, tok in enumerate(tokens):
            if isinstance(tok, ConstantToken):
                const_idx.append(i)
                consts.append(tok.constant)
                continue
            emb_idx.append(i)
            if isinstance(tok, SequenceIndicatorToken):
                emb_type_idx.append(int(tok.indicator))
            elif isinstance(tok, FeatureToken):
                emb_type_idx.append(int(tok.feature) + feat_offset)
            elif isinstance(tok, OperatorToken):
                emb_type_idx.append(Operators.index(tok.operator) + op_offset)
            elif isinstance(tok, DeltaTimeToken):
                emb_type_idx.append(int(tok.delta_time) - self._delta_time_range[0] + dt_offset)
            else:
                assert False, "NullToken is not allowed here"

        result = torch.zeros(len(tokens), self._d_model,
                             dtype=torch.float, device=self._device)
        if len(const_idx) != 0:
            const_tensor = torch.tensor(consts, device=self._device).unsqueeze(1)
            result[const_idx] = self._const_linear(const_tensor)
        if len(emb_idx) != 0:
            result[emb_idx] = self._emb(torch.tensor(emb_type_idx, dtype=torch.long, device=self._device))

        return result


# Deprecated!
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('_pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        "x: ([batch_size, ]seq_len, embedding_dim)"
        seq_len = x.size(0) if x.dim() == 2 else x.size(1)
        return x + self._pe[:seq_len]  # type: ignore


# Deprecated!
class ExpressionGenerator(nn.Module):
    def __init__(self,
                 n_encoder_layers: int,
                 n_decoder_layers: int,
                 d_model: int,
                 n_head: int,
                 d_ffn: int,
                 dropout: float,
                 operators: List[Type[Operator]],
                 delta_time_range: Tuple[int, int],
                 device: torch.device):
        super().__init__()
        self._device = device
        self._d_model = d_model
        self._operators = operators
        self._dt_range = delta_time_range

        self._token_emb = TokenEmbedding(d_model, operators, delta_time_range, device)
        self._pos_enc = PositionalEncoding(d_model).to(device)
        self._encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head, dim_feedforward=d_ffn,
                dropout=dropout, device=device
            ),
            n_encoder_layers,
            norm=nn.LayerNorm(d_model, device=device)
        )
        self._decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=n_head, dim_feedforward=d_ffn,
                dropout=dropout, device=device
            ),
            n_decoder_layers,
            norm=nn.LayerNorm(d_model, device=device)
        )
        self._op_feat_const_linear = nn.Linear(d_model, 3, device=device)
        self._op_linear = nn.Linear(d_model, len(operators), device=device)
        self._feat_linear = nn.Linear(d_model, len(FeatureType), device=device)
        self._const_linear = nn.Linear(d_model, 2, device=device)
        self._dt_linear = nn.Linear(d_model, delta_time_range[1] - delta_time_range[0],
                                    device=device)

    def embed_expressions(self, expr: List[Token]) -> Tensor:
        res: Tensor = self._token_emb(expr)
        return self._pos_enc(res)

    def encode_expressions(self, expr: List[Token]) -> Tensor:
        res = self.embed_expressions(expr)
        res = self._encoder.forward(res.unsqueeze(1))
        return res.squeeze(1)

    def forward(
            self,
            encoder_state: Tensor,
            decoder_tokens: List[Token],
            sample_delta_time: bool = False) -> Tuple[Token, Tensor]:
        """
            Sample the next token.
            Returns the token and its corresponding log-prob (normalized)
        """
        decoder_tokens = decoder_tokens.copy()
        decoder_tokens.insert(0, SEP_TOKEN)
        length = len(decoder_tokens)
        causal_mask = torch.triu(  # [L, L]
            torch.ones(length, length, dtype=torch.bool, device=self._device),
            diagonal=1)
        decoder_state = self.embed_expressions(decoder_tokens)
        res: Tensor = self._decoder(  # [L, 1, D]
            decoder_state.unsqueeze(1),
            encoder_state.unsqueeze(1),
            tgt_mask=causal_mask
        )
        res = res.mean(dim=0).reshape(-1)  # [D]

        if sample_delta_time:
            dist = Categorical(logits=self._dt_linear(res))
            idx = dist.sample()
            log_prob = dist.log_prob(idx)
            dt = self._dt_range[0] + int(idx)
            return DeltaTimeToken(dt), log_prob

        select_dist = Categorical(logits=self._op_feat_const_linear(res))
        idx = select_dist.sample()
        select_log_prob = select_dist.log_prob(idx)
        idx = int(idx)
        if idx == 0:  # Operators
            dist = Categorical(logits=self._op_linear(res))
            idx = dist.sample()
            log_prob = select_log_prob + dist.log_prob(idx)
            return OperatorToken(self._operators[int(idx)]), log_prob
        elif idx == 1:  # Features
            dist = Categorical(logits=self._feat_linear(res))
            idx = dist.sample()
            log_prob = select_log_prob + dist.log_prob(idx)
            return FeatureToken(FeatureType(int(idx))), log_prob
        else:  # Constants
            affine: Tensor = self._const_linear(res)
            mu, sigma = affine[0], affine[1].exp()
            dist = Normal(mu, sigma)
            z = dist.sample()
            log_prob = select_log_prob + dist.log_prob(z)
            return ConstantToken(float(z)), log_prob
