from typing import List
from alphagen.data.exception import InvalidExpressionException
from alphagen.data.expression import BinaryOperator, Constant, DeltaTime, Expression, Feature, PairRollingOperator, RollingOperator, UnaryOperator
from alphagen.data.tokens import *


class ExpressionBuilder:
    stack: List[Expression]

    def __init__(self):
        self.stack = []

    def get_tree(self) -> Expression:
        if len(self.stack) == 1:
            return self.stack[0]
        else:
            raise InvalidExpressionException(f"Expected only one tree, got {len(self.stack)}")

    def add_token(self, token: Token):
        if not self.validate(token):
            raise InvalidExpressionException(f"Token {token} not allowed here, stack: {self.stack}.")
        if isinstance(token, OperatorToken):
            n_args: int = token.operator.n_args()
            children = []
            for _ in range(n_args):
                children.append(self.stack.pop())
            self.stack.append(token.operator(*reversed(children)))  # type: ignore
        elif isinstance(token, ConstantToken):
            self.stack.append(Constant(token.constant))
        elif isinstance(token, DeltaTimeToken):
            self.stack.append(DeltaTime(token.delta_time))
        elif isinstance(token, FeatureToken):
            self.stack.append(Feature(token.feature))
        elif isinstance(token, ExpressionToken):
            self.stack.append(token.expression)
        else:
            assert False

    def is_valid(self) -> bool:
        return len(self.stack) == 1 and self.stack[0].is_featured

    def validate(self, token: Token) -> bool:
        if isinstance(token, OperatorToken):
            return self.validate_op(token.operator)
        elif isinstance(token, DeltaTimeToken):
            return self.validate_dt()
        elif isinstance(token, ConstantToken):
            return self.validate_const()
        elif isinstance(token, (FeatureToken, ExpressionToken)):
            return self.validate_featured_expr()
        else:
            assert False

    def validate_op(self, op: Type[Operator]) -> bool:
        if len(self.stack) < op.n_args():
            return False

        if issubclass(op, UnaryOperator):
            if not self.stack[-1].is_featured:
                return False
        elif issubclass(op, BinaryOperator):
            if not self.stack[-1].is_featured and not self.stack[-2].is_featured:
                return False
            if (isinstance(self.stack[-1], DeltaTime) or
                    isinstance(self.stack[-2], DeltaTime)):
                return False
        elif issubclass(op, RollingOperator):
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if not self.stack[-2].is_featured:
                return False
        elif issubclass(op, PairRollingOperator):
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if not self.stack[-2].is_featured or not self.stack[-3].is_featured:
                return False
        else:
            assert False
        return True

    def validate_dt(self) -> bool:
        return len(self.stack) > 0 and self.stack[-1].is_featured

    def validate_const(self) -> bool:
        return len(self.stack) == 0 or self.stack[-1].is_featured

    def validate_featured_expr(self) -> bool:
        return not (len(self.stack) >= 1 and isinstance(self.stack[-1], DeltaTime))
