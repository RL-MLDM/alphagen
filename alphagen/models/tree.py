from typing import Union

from alphagen.models.tokens import *


class AlphaTree:
    value: Union[OperatorToken, ConstantToken, DeltaTimeToken, FeatureToken]

    # "Featured" means tree is composed of some features
    # Constant alpha may cause qlib exception
    is_featured: bool = False

    def __repr__(self):
        return str(self)


class OperatorTreeNode(AlphaTree):
    children_reversed: List['AlphaTree']

    def __init__(self, operator: OperatorToken):
        self.value = operator
        self.children_reversed = []

    def append_child(self, child: 'AlphaTree') -> 'AlphaTree':
        self.children_reversed.append(child)
        if child.is_featured:
            self.is_featured = True
        return self

    def __str__(self):
        params = ','.join([str(child) for child in reversed(self.children_reversed)])
        return f'{self.value}({params})'


class ValueTreeNode(AlphaTree):
    value: Union[ConstantToken, DeltaTimeToken, FeatureToken]

    def __init__(self, value: Union[ConstantToken, DeltaTimeToken, FeatureToken]):
        self.value = value
        if isinstance(value, FeatureToken):
            self.is_featured = True

    def __str__(self):
        return str(self.value)


class AlphaTreeBuilder:
    stack: List[AlphaTree] = []

    def __init__(self):
        self.stack = []

    def get_tree(self) -> AlphaTree:
        if len(self.stack) == 1:
            return self.stack[0]
        else:
            raise InvalidTreeException(f'Expected only one tree, got {len(self.stack)}')

    def add_token(self, token: Token):
        if not self.validate(token):
            raise InvalidTreeException(f'Token {token} not allowed here, stack {self.stack}.')
        if isinstance(token, OperatorToken):
            n_args = token.operator.category.n_args
            node = OperatorTreeNode(token)
            for _ in range(n_args):
                node.append_child(self.stack.pop())     # Last argument first in
            self.stack.append(node)
        else:
            self.stack.append(ValueTreeNode(token))     # type: ignore

    def is_valid(self) -> bool:
        return len(self.stack) == 1 and self.stack[0].is_featured

    def validate(self, token: Token) -> bool:
        if isinstance(token, OperatorToken):
            return self.validate_op(token.operator.category)
        elif isinstance(token, DeltaTimeToken):
            return self.validate_dt()
        elif isinstance(token, ConstantToken):
            return self.validate_const()
        elif isinstance(token, FeatureToken):
            return self.validate_feature()
        else:
            assert False

    def validate_op(self, op_category: OperatorCategory) -> bool:
        n_args = op_category.n_args
        if len(self.stack) < n_args:
            return False

        if op_category == OperatorCategory.UNARY:
            if not self.stack[-1].is_featured:
                return False
        elif op_category == OperatorCategory.BINARY:
            if not self.stack[-1].is_featured and not self.stack[-2].is_featured:
                return False
            if isinstance(self.stack[-1].value, DeltaTimeToken) or isinstance(self.stack[-2].value, DeltaTimeToken):
                return False
        elif op_category == OperatorCategory.ROLLING:
            if not isinstance(self.stack[-1].value, DeltaTimeToken):
                return False
            if not self.stack[-2].is_featured:
                return False
        elif op_category == OperatorCategory.BINARY_ROLLING:
            if not isinstance(self.stack[-1].value, DeltaTimeToken):
                return False
            if not self.stack[-2].is_featured or not self.stack[-3].is_featured:
                return False
        else:
            assert False
        return True

    def validate_dt(self) -> bool:
        if len(self.stack) < 1 or not self.stack[-1].is_featured:
            return False
        else:
            return True

    def validate_const(self) -> bool:
        if len(self.stack) >= 1 and not self.stack[-1].is_featured:
            return False
        else:
            return True

    def validate_feature(self) -> bool:
        if len(self.stack) >= 1 and isinstance(self.stack[-1].value, DeltaTimeToken):
            return False
        else:
            return True


class InvalidTreeException(ValueError):
    pass


if __name__ == '__main__':
    tokens = [
        FeatureToken(FeatureType.LOW),
        OperatorToken(OperatorType.ABS),
        DeltaTimeToken(-10),
        OperatorToken(OperatorType.REF),
        FeatureToken(FeatureType.HIGH),
        FeatureToken(FeatureType.CLOSE),
        OperatorToken(OperatorType.DIV),
        OperatorToken(OperatorType.ADD),
    ]

    builder = AlphaTreeBuilder()
    for token in tokens:
        builder.add_token(token)

    print(f'res: {str(builder.get_tree())}')
    print(f'ref: Add(Ref(Abs($low),-10),Div($high,$close))')
