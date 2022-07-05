from typing import Tuple
import gym

from alphagen.data.evaluation import Evaluation
from alphagen.models.tokens import *
from alphagen.models.tree import AlphaTreeBuilder

MAX_TOKEN_LENGTH = 15


class AlphaEnvCore(gym.Env):
    ev: Evaluation
    tokens: List[Token]
    builder: AlphaTreeBuilder

    def __init__(self, instrument: str, start_time: str, end_time: str):
        super().__init__()
        self.ev = Evaluation(instrument, start_time, end_time)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Tuple[List[Token], dict]:
        self.tokens = [SequenceIndicatorToken(SequenceIndicatorType.BEG)]
        self.builder = AlphaTreeBuilder()
        return self.tokens, self._valid_action_types()

    def step(self, action: Token) -> Tuple[List[Token], float, bool, dict]:
        if isinstance(action, SequenceIndicatorToken) and action.indicator == SequenceIndicatorType.SEP:
            done = True
            reward = self._evaluate()
        elif len(self.tokens) < MAX_TOKEN_LENGTH:
            self.tokens.append(action)
            self.builder.add_token(action)
            done = False
            reward = 0.0
        else:
            done = True
            reward = self._evaluate() if self.builder.is_valid() else -1.0
        return self.tokens, reward, done, self._valid_action_types()

    def _evaluate(self):
        expr: str = str(self.builder.get_tree())
        return self.ev.evaluate(expr)

    def _valid_action_types(self) -> dict:
        valid_op_unary = self.builder.validate_op(OperatorCategory.UNARY)
        valid_op_binary = self.builder.validate_op(OperatorCategory.BINARY)
        valid_op_rolling = self.builder.validate_op(OperatorCategory.ROLLING)
        valid_op_binary_rolling = False  # self.builder.validate_op(OperatorCategory.BINARY_ROLLING)

        valid_op = valid_op_unary or valid_op_binary or valid_op_rolling or valid_op_binary_rolling
        valid_dt = self.builder.validate_dt()
        valid_const = self.builder.validate_const()
        valid_feature = self.builder.validate_feature()
        valid_stop = self.builder.is_valid()

        ret = {
            'select': [valid_op, valid_feature, valid_const, valid_dt, valid_stop],
            'op': [valid_op_unary, valid_op_binary, valid_op_rolling, valid_op_binary_rolling]
        }
        return ret

    def render(self, mode="human"):
        pass


if __name__ == '__main__':
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    env = AlphaEnvCore(
        instrument='csi300',
        start_time='2016-01-01',
        end_time='2018-12-31'
    )

    from alphagen.models.tokens import *
    tokens = [
        FeatureToken(FeatureType.LOW),
        OperatorToken(OperatorType.ABS),
        DeltaTimeToken(-10),
        OperatorToken(OperatorType.REF),
        FeatureToken(FeatureType.HIGH),
        FeatureToken(FeatureType.CLOSE),
        OperatorToken(OperatorType.DIV),
        OperatorToken(OperatorType.ADD),
        SEP_TOKEN,
    ]

    state = env.reset()
    for token in tokens:
        state, reward, done, info = env.step(token)
        print(f'next_state: {state}, reward: {reward}, done: {done}, info: {info}')


