from abc import ABCMeta, abstractmethod
from enum import IntEnum
from typing import List, Optional


class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    # VWAP = 5

    ENUM_SIZE = 5


class OperatorCategory(IntEnum):
    UNARY = 0
    BINARY = 1
    ROLLING = 2
    BINARY_ROLLING = 3

    @property
    def n_args(self) -> int:
        if self == OperatorCategory.UNARY:
            return 1
        elif self == OperatorCategory.BINARY or self == OperatorCategory.ROLLING:
            return 2
        elif self == OperatorCategory.BINARY_ROLLING:
            return 3
        else:
            raise ValueError("Invalid operator")


class OperatorType(IntEnum):
    # Unary
    ABS = 0
    SIGN = 1
    LOG = 2

    # Binary
    ADD = 3
    SUB = 4
    MUL = 5
    DIV = 6
    GREATER = 7
    LESS = 8

    # Rolling
    REF = 9
    MEAN = 10
    SUM = 11
    STD = 12
    VAR = 13
    SKEW = 14
    KURT = 15
    MAX = 16
    MIN = 17
    QUANTILE = 18
    MED = 19
    MAD = 20
    RANK = 21
    DELTA = 22
    RSQUARE = 23
    RESI = 24
    WMA = 25
    EMA = 26

    # Binary rolling
    CORR = 27
    COV = 28

    ENUM_SIZE = 29

    @property
    def category(self) -> OperatorCategory:
        val = int(self)
        if val <= 2:
            return OperatorCategory.UNARY
        elif val <= 8:
            return OperatorCategory.BINARY
        elif val <= 26:
            return OperatorCategory.ROLLING
        elif val <= 28:
            return OperatorCategory.BINARY_ROLLING
        else:
            raise ValueError("Invalid operator")

    @property
    def operator_name(self) -> str:
        if self in [OperatorType.EMA, OperatorType.WMA]:
            return self.name
        return self.name.capitalize()


class SequenceIndicatorType(IntEnum):
    BEG = 0
    SEP = 1

    ENUM_SIZE = 2


class Token:
    def __repr__(self):
        return str(self)


class ConstantToken(Token):
    def __init__(self, constant: float) -> None:
        self.constant = constant

    def __str__(self):
        return str(self.constant)


class DeltaTimeToken(Token):
    def __init__(self, delta_time: int) -> None:
        self.delta_time = delta_time

    def __str__(self):
        return str(self.delta_time)


class FeatureToken(Token):
    def __init__(self, feature: FeatureType) -> None:
        self.feature = feature

    def __str__(self):
        return '$' + self.feature.name.lower()


class OperatorToken(Token):
    def __init__(self, operator: OperatorType) -> None:
        self.operator = operator

    def __str__(self):
        return self.operator.operator_name


class SequenceIndicatorToken(Token):
    def __init__(self, indicator: SequenceIndicatorType) -> None:
        self.indicator = indicator

    def __str__(self):
        return self.indicator.name


BEG_TOKEN = SequenceIndicatorToken(SequenceIndicatorType.BEG)
SEP_TOKEN = SequenceIndicatorToken(SequenceIndicatorType.SEP)
