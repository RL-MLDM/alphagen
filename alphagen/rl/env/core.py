from typing import Tuple, Optional
import gym
import math

from alphagen.config import MAX_EXPR_LENGTH
from alphagen.data.evaluation import Evaluation
from alphagen.data.tokens import *
from alphagen.data.expression import *
from alphagen.data.tree import ExpressionBuilder
from alphagen.utils import reseed_everything, batch_spearmanr


class AlphaEnvCore(gym.Env):
    _eval: Evaluation
    _tokens: List[Token]
    _builder: ExpressionBuilder
    _record_path: Optional[str]

    def __init__(
        self,
        instrument: str,
        start_time: str, end_time: str,
        *,
        record_path: Optional[str] = None,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()

        close = Feature(FeatureType.CLOSE)
        target = Ref(close, -20) / close - 1

        inst = StockData.list_instruments(instrument, start_time, end_time)
        self._eval = Evaluation(inst, start_time, end_time, target, device)
        self._record_path = record_path
        self._device = device

    def reset(
        self, *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Tuple[List[Token], dict]:
        reseed_everything(seed)
        self._exprs = []
        self._tokens = [BEG_TOKEN]
        self._builder = ExpressionBuilder()
        return self._tokens, self._valid_action_types()

    def step(self, action: Token) -> Tuple[List[Token], float, bool, dict]:
        if (isinstance(action, SequenceIndicatorToken) and
                action.indicator == SequenceIndicatorType.SEP):
            reward = self._evaluate()
            done = True
        elif len(self._tokens) < MAX_EXPR_LENGTH:
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

    def _maybe_record_expr(self, expr: Expression, score: float):
        if self._record_path is not None:
            with open(self._record_path, "a") as f:
                f.write(f"{score}\t{expr}\n")

    def _evaluate(self) -> float:
        expr: Expression = self._builder.get_tree()
        score = self._eval.evaluate(expr)
        self._maybe_record_expr(expr, score)
        return score

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
