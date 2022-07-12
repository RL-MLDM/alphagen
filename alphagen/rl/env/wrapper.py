import gym
import numpy as np

from sb3_contrib.common.wrappers import ActionMasker

from alphagen.config import *
from alphagen.data.tokens import *
from alphagen.rl.env.core import AlphaEnvCore

SIZE_NULL = 1
SIZE_OP = len(OPERATORS)
SIZE_FEATURE = len(FeatureType)
SIZE_DELTA_TIME = len(DELTA_TIMES)
SIZE_CONSTANT = len(CONSTANTS)
SIZE_SEP = 1

SIZE_ALL = SIZE_NULL + SIZE_OP + SIZE_FEATURE + SIZE_DELTA_TIME + SIZE_CONSTANT + SIZE_SEP
SIZE_ACTION = SIZE_ALL - SIZE_NULL

OFFSET_OP = SIZE_NULL
OFFSET_FEATURE = OFFSET_OP + SIZE_OP
OFFSET_DELTA_TIME = OFFSET_FEATURE + SIZE_FEATURE
OFFSET_CONSTANT = OFFSET_DELTA_TIME + SIZE_DELTA_TIME
OFFSET_SEP = OFFSET_CONSTANT + SIZE_CONSTANT


def action2token(action_raw: int) -> Token:
    action = action_raw + 1
    if action < OFFSET_OP:
        raise ValueError
    elif action < OFFSET_FEATURE:
        return OperatorToken(OPERATORS[action - OFFSET_OP])
    elif action < OFFSET_DELTA_TIME:
        return FeatureToken(FeatureType(action - OFFSET_FEATURE))
    elif action < OFFSET_CONSTANT:
        return DeltaTimeToken(DELTA_TIMES[action - OFFSET_DELTA_TIME])
    elif action < OFFSET_SEP:
        return ConstantToken(CONSTANTS[action - OFFSET_CONSTANT])
    elif action == OFFSET_SEP:
        return SequenceIndicatorToken(SequenceIndicatorType.SEP)
    else:
        assert False


class AlphaEnvWrapper(gym.Wrapper):
    counter: int
    state: np.ndarray

    def __init__(self, env: AlphaEnvCore):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(SIZE_ACTION)
        self.observation_space = gym.spaces.Box(low=0, high=SIZE_ALL-1, shape=(MAX_TOKEN_LENGTH, ), dtype=np.uint8)

    def reset(self, **kwargs) -> np.ndarray:
        self.counter = 0
        self.state = np.zeros(MAX_TOKEN_LENGTH, dtype=np.uint8)
        self.env.reset()
        return self.state

    def step(self, action: int):
        observation, reward, done, info = self.env.step(self.action(action))
        if not done:
            self.state[self.counter] = action + 1
            self.counter += 1
        return self.state, self.reward(reward), done, info

    def action(self, action: int) -> Token:
        return action2token(action)

    def reward(self, reward: float) -> float:
        return reward + REWARD_PER_STEP

    def valid_action_mask(self) -> np.ndarray:
        res = np.zeros(SIZE_ACTION, dtype=bool)
        valid = self.env.valid_action_types()
        for i in range(OFFSET_OP, OFFSET_OP + SIZE_OP):
            if valid['op'][OPERATORS[i - OFFSET_OP].category_type()]:
                res[i-1] = True
        if valid['select'][1]:  # FEATURE
            for i in range(OFFSET_FEATURE, OFFSET_FEATURE + SIZE_FEATURE):
                res[i-1] = True
        if valid['select'][2]:  # CONSTANT
            for i in range(OFFSET_CONSTANT, OFFSET_CONSTANT + SIZE_CONSTANT):
                res[i-1] = True
        if valid['select'][3]:  # DELTA_TIME
            for i in range(OFFSET_DELTA_TIME, OFFSET_DELTA_TIME + SIZE_DELTA_TIME):
                res[i-1] = True
        if valid['select'][4]:  # SEP
            res[OFFSET_SEP-1] = True
        return res


def AlphaEnv(*args, **kwargs):
    return ActionMasker(AlphaEnvWrapper(AlphaEnvCore(*args, **kwargs)), lambda env: env.valid_action_mask())


if __name__ == '__main__':
    env = AlphaEnv(
        instrument="csi300",
        start_time="2016-01-01",
        end_time="2018-12-31"
    )

    state = env.reset()
    actions = [
        OFFSET_FEATURE + FeatureType.LOW,
        OFFSET_OP + 5,  # Abs
        OFFSET_DELTA_TIME + 1,
        OFFSET_OP + 4,  # Ref
        OFFSET_FEATURE + FeatureType.HIGH,
        OFFSET_FEATURE + FeatureType.CLOSE,
        OFFSET_OP + 3,  # Div
        OFFSET_OP + 0,  # Add
        OFFSET_SEP,
    ]
    for action in actions:
        print(env.step(action)[:-1])