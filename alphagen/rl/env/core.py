from typing import Tuple, Optional
import gym
import math

from alphagen.config import MAX_TOKEN_LENGTH
from alphagen.data.evaluation import Evaluation
from alphagen.data.tokens import *
from alphagen.data.expression import *
from alphagen.data.tree import ExpressionBuilder
from alphagen.utils import reseed_everything, batch_spearmanr


class AlphaEnvCore(gym.Env):
    _eval: Evaluation
    _max_expressions: int
    _exprs: List[Expression]
    _tokens: List[Token]
    _builder: ExpressionBuilder

    def __init__(self,
                 instrument: str,
                 start_time: str, end_time: str,
                 *,
                 max_expressions: int = 5,
                 device: torch.device = torch.device("cpu")):
        super().__init__()

        close = Feature(FeatureType.CLOSE)
        target = Ref(close, -20) / close - 1

        self._eval = Evaluation(instrument, start_time, end_time, target, device)
        self._device = device
        self._max_expressions = max_expressions

    def reset(self, *,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None) -> Tuple[List[Token], dict]:
        reseed_everything(seed)
        self._exprs = []
        self._tokens = [BEG_TOKEN]
        self._builder = ExpressionBuilder()
        return self._tokens, self._valid_action_types()

    def step(self, action: Token) -> Tuple[List[Token], float, bool, dict]:
        if (isinstance(action, SequenceIndicatorToken) and
                action.indicator == SequenceIndicatorType.SEP):
            reward = self._evaluate()
            done = reward < 0 or len(self._exprs) == self._max_expressions
            self._tokens = [BEG_TOKEN]
            self._builder = ExpressionBuilder()
        elif len(self._tokens) < MAX_TOKEN_LENGTH:
            self._tokens.append(action)
            self._builder.add_token(action)
            done = False
            reward = 0.0
        else:
            done = True
            reward = self._evaluate() if self._builder.is_valid() else -1.0
        if math.isnan(reward):
            reward = -1
        return self._tokens, reward, done, self._valid_action_types()

    def _evaluate(self) -> float:
        data = self._eval.data
        expr = self._builder.get_tree()
        ic = self._eval.evaluate(expr)
        max_corr = 0.
        for e in self._exprs:
            try:
                corrs = batch_spearmanr(expr.evaluate(data), e.evaluate(data))
            except OutOfDataRangeError:
                continue
            corr = corrs.mean().item()
            max_corr = max(max_corr, corr)
        discount = 1 if ic <= 0 else 1 - max_corr
        self._exprs.append(expr)
        return ic * discount

    def _valid_action_types(self) -> dict:
        valid_op_unary = self._builder.validate_op(UnaryOperator)
        valid_op_binary = self._builder.validate_op(BinaryOperator)
        valid_op_rolling = self._builder.validate_op(RollingOperator)
        valid_op_pair_rolling = False  # self.builder.validate_op(PairRollingOperator)

        valid_op = valid_op_unary or valid_op_binary or valid_op_rolling or valid_op_pair_rolling
        valid_dt = self._builder.validate_dt()
        valid_const = self._builder.validate_const()
        valid_feature = self._builder.validate_feature()
        valid_stop = self._builder.is_valid()

        ret = {
            "select": [valid_op, valid_feature, valid_const, valid_dt, valid_stop],
            "op": {
                UnaryOperator: valid_op_unary,
                BinaryOperator: valid_op_binary,
                RollingOperator: valid_op_rolling,
                PairRollingOperator: valid_op_pair_rolling
            }
        }
        return ret

    def valid_action_types(self) -> dict:
        return self._valid_action_types()

    def render(self, mode="human"):
        pass


if __name__ == '__main__':
    env = AlphaEnvCore(
        instrument='csi300',
        start_time='2016-01-01',
        end_time='2018-12-31'
    )

    tokens = [
        FeatureToken(FeatureType.LOW),
        OperatorToken(Abs),
        DeltaTimeToken(10),
        OperatorToken(Ref),
        FeatureToken(FeatureType.HIGH),
        FeatureToken(FeatureType.CLOSE),
        OperatorToken(Div),
        OperatorToken(Add),
        SEP_TOKEN,
    ]

    state = env.reset()
    for token in tokens:
        state, reward, done, info = env.step(token)
        print(f'next_state: {state}, reward: {reward}, done: {done}, info: {info}')
