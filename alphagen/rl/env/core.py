from typing import Tuple, Optional
import gym
import math

from alphagen.config import MAX_TOKEN_LENGTH
from alphagen.data.evaluation import EvaluationBase
from alphagen.data.tokens import *
from alphagen.data.expression import *
from alphagen.data.tree import AlphaTreeBuilder
from alphagen.utils.random import reseed_everything


class AlphaEnvCore(gym.Env):
    _ev: EvaluationBase
    _tokens: List[Token]
    _builder: AlphaTreeBuilder
    _print_expr: bool
    _dirty_action: bool

    def __init__(self,
                 ev: EvaluationBase,
                 print_expr: bool = False,
                 dirty_action: bool = False):
        super().__init__()

        self._ev = ev
        self._print_expr = print_expr
        self._dirty_action = dirty_action

    def reset(self, *,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None) -> Tuple[List[Token], dict]:
        reseed_everything(seed)
        self._tokens = [SequenceIndicatorToken(SequenceIndicatorType.BEG)]
        self._builder = AlphaTreeBuilder()
        return self._tokens, self._valid_action_types()

    def step(self, action: Token) -> Tuple[List[Token], float, bool, dict]:
        if (isinstance(action, SequenceIndicatorToken) and
                action.indicator == SequenceIndicatorType.SEP):
            done = True
            reward = self._evaluate() if self._builder.is_valid() else -1.0
        elif self._dirty_action and not self._builder.validate(action):
            done = True
            reward = -1
        else:
            self._tokens.append(action)
            self._builder.add_token(action)
            if len(self._tokens) >= MAX_TOKEN_LENGTH:
                done = True
                reward = self._evaluate() if self._builder.is_valid() else -1.0
            else:
                done = False
                reward = 0.0

        if math.isnan(reward):
            reward = -1
        return self._tokens, reward, done, self._valid_action_types()

    def _evaluate(self):
        expr: Expression = self._builder.get_tree()
        if self._print_expr:
            print(expr)
        return self._ev.evaluate(expr)

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
    from alphagen.data.evaluation import QLibEvaluation

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    ev = QLibEvaluation('csi300', '2016-01-01', '2018-12-31', target)
    env = AlphaEnvCore(ev=ev)

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
