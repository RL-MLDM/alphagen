from typing import List, Type
from alphagen.data.expression import *


MAX_EXPR_LENGTH = 20
MAX_EPISODE_LENGTH = 256

OPERATORS: List[Type[Operator]] = [Add, Sub, Mul, Div, Ref, Abs, EMA, Sum, Mean, Std]
DELTA_TIMES = [2, 10, 20, 30, 40, 50]
CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]

REWARD_PER_STEP = 0.0  # 0.00005
