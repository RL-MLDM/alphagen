from alphagen.data.expression import Expression


class Evaluation:
    def evaluate(self, expr: Expression) -> float:
        raise NotImplementedError
