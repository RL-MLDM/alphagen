from alphagen.data.expression import *


MAX_EXPR_LENGTH = 20
MAX_EPISODE_LENGTH = 256

OPERATORS = [
    # Unary
    Abs,  # Sign,
    Log,
    # Binary
    Add, Sub, Mul, Div, Greater, Less,
    # Rolling
    Ref, Mean, Sum, Std, Var,  # Skew, Kurt,
    Max, Min,
    Med, Mad,  # Rank,
    Delta, WMA, EMA,
    # Pair rolling
    Cov, Corr
]

DELTA_TIMES = [10, 20, 30, 40, 50]

CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]

REWARD_PER_STEP = 0.
