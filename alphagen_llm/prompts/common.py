from enum import IntEnum
from typing import Optional, List, Tuple
from num2words import num2words
from alphagen.data.expression import Expression
from alphagen.data.parser import ExpressionParser


class MetricDescriptionMode(IntEnum):
    NOT_INCLUDED = 0    # Description of this metric is not included in the prompt.
    INCLUDED = 1        # Description of this metric is included in the prompt.
    SORTED_BY = 2       # Description is included, and the alphas will be sorted according to this metric.


def alpha_word(n: int) -> str: return "alpha" if n == 1 else "alphas"


def alpha_phrase(n: int, adjective: Optional[str] = None) -> str:
    n_word = str(n) if n > 10 else num2words(n)
    adjective = f" {adjective}" if adjective is not None else ""
    return f"{n_word}{adjective} {alpha_word(n)}"


def safe_parse(parser: ExpressionParser, expr_str: str) -> Optional[Expression]:
    try:
        return parser.parse(expr_str)
    except:
        return None


def safe_parse_list(lines: List[str], parser: ExpressionParser) -> Tuple[List[Expression], List[str]]:
    parsed, invalid = [], []
    for line in lines:
        if line == "":
            continue
        if (e := safe_parse(parser, line)) is not None:
            parsed.append(e)
        else:
            invalid.append(line)
    return parsed, invalid
