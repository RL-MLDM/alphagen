from typing import Tuple, Optional
import gym
import math

from alphagen.config import MAX_EXPR_LENGTH
from alphagen.data.evaluation import Evaluation
from alphagen.data.tokens import *
from alphagen.data.expression import *
from alphagen.data.tree import ExpressionBuilder
from alphagen.utils import reseed_everything


class AlphaEnvCore(gym.Env):
    eval: Evaluation
    _tokens: List[Token]
    _builder: ExpressionBuilder
    _print_expr: bool

    def __init__(self,
                 ev: Evaluation,
                 device: torch.device = torch.device("cpu"),
                 print_expr: bool = False):
        super().__init__()

        self.eval = ev
        self._print_expr = print_expr
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

    def _evaluate(self):
        expr: Expression = self._builder.get_tree()
        if self._print_expr:
            print(expr)
        return self.eval.evaluate(expr)

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
    from alphagen_qlib.evaluation import QLibEvaluation

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    ev = QLibEvaluation(
        instrument='csi300',
        start_time='2016-01-01',
        end_time='2018-12-31',
        target=target
    )
    env = AlphaEnvCore(ev)

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
