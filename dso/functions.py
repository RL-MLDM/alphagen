"""Common Tokens used for executable Programs."""
import re
import numpy as np
from fractions import Fraction

from dso.library import Token, PlaceholderConstant, HardCodedConstant, Polynomial, StateChecker
import dso.utils as U

GAMMA = 0.57721566490153286060651209008240243104215933593992


"""Define custom unprotected operators"""
def logabs(x1):
    """Closure of log for non-positive arguments."""
    return np.log(np.abs(x1))

def expneg(x1):
    return np.exp(-x1)

def n3(x1):
    return np.power(x1, 3)

def n4(x1):
    return np.power(x1, 4)

def sigmoid(x1):
    return 1 / (1 + np.exp(-x1))

def harmonic(x1):
    if all(val.is_integer() for val in x1):
        return np.array([sum(Fraction(1, d) for d in range(1, int(val)+1)) for val in x1], dtype=np.float32)
    else:
        return GAMMA + np.log(x1) + 0.5/x1 - 1./(12*x1**2) + 1./(120*x1**4)


# Annotate unprotected ops
unprotected_ops = [
    # Binary operators
    Token(np.add, "add", arity=2, complexity=1),
    Token(np.subtract, "sub", arity=2, complexity=1),
    Token(np.multiply, "mul", arity=2, complexity=1),
    Token(np.divide, "div", arity=2, complexity=2),

    # Built-in unary operators
    Token(np.sin, "sin", arity=1, complexity=3),
    Token(np.cos, "cos", arity=1, complexity=3),
    Token(np.tan, "tan", arity=1, complexity=4),
    Token(np.exp, "exp", arity=1, complexity=4),
    Token(np.log, "log", arity=1, complexity=4),
    Token(np.sqrt, "sqrt", arity=1, complexity=4),
    Token(np.square, "n2", arity=1, complexity=2),
    Token(np.negative, "neg", arity=1, complexity=1),
    Token(np.abs, "abs", arity=1, complexity=2),
    Token(np.maximum, "max", arity=1, complexity=4),
    Token(np.minimum, "min", arity=1, complexity=4),
    Token(np.tanh, "tanh", arity=1, complexity=4),
    Token(np.reciprocal, "inv", arity=1, complexity=2),

    # Custom unary operators
    Token(logabs, "logabs", arity=1, complexity=4),
    Token(expneg, "expneg", arity=1, complexity=4),
    Token(n3, "n3", arity=1, complexity=3),
    Token(n4, "n4", arity=1, complexity=3),
    Token(sigmoid, "sigmoid", arity=1, complexity=4),
    Token(harmonic, "harmonic", arity=1, complexity=4)
]


"""Define custom protected operators"""
def protected_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)

def protected_exp(x1):
    with np.errstate(over='ignore'):
        return np.where(x1 < 100, np.exp(x1), 0.0)

def protected_log(x1):
    """Closure of log for non-positive arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)

def protected_sqrt(x1):
    """Closure of sqrt for negative arguments."""
    return np.sqrt(np.abs(x1))

def protected_inv(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)

def protected_expneg(x1):
    with np.errstate(over='ignore'):
        return np.where(x1 > -100, np.exp(-x1), 0.0)

def protected_n2(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.square(x1), 0.0)

def protected_n3(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.power(x1, 3), 0.0)

def protected_n4(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.power(x1, 4), 0.0)

def protected_sigmoid(x1):
    return 1 / (1 + protected_expneg(x1))

# Annotate protected ops
protected_ops = [
    # Protected binary operators
    Token(protected_div, "div", arity=2, complexity=2),

    # Protected unary operators
    Token(protected_exp, "exp", arity=1, complexity=4),
    Token(protected_log, "log", arity=1, complexity=4),
    Token(protected_log, "logabs", arity=1, complexity=4), # Protected logabs is support, but redundant
    Token(protected_sqrt, "sqrt", arity=1, complexity=4),
    Token(protected_inv, "inv", arity=1, complexity=2),
    Token(protected_expneg, "expneg", arity=1, complexity=4),
    Token(protected_n2, "n2", arity=1, complexity=2),
    Token(protected_n3, "n3", arity=1, complexity=3),
    Token(protected_n4, "n4", arity=1, complexity=3),
    Token(protected_sigmoid, "sigmoid", arity=1, complexity=4)
]

# Add unprotected ops to function map
function_map = {
    op.name : op for op in unprotected_ops
    }

# Add protected ops to function map
function_map.update({
    "protected_{}".format(op.name) : op for op in protected_ops
    })

TERMINAL_TOKENS = set([op.name for op in function_map.values() if op.arity == 0])
UNARY_TOKENS    = set([op.name for op in function_map.values() if op.arity == 1])
BINARY_TOKENS   = set([op.name for op in function_map.values() if op.arity == 2])


def create_state_checkers(n_states, threshold_set):
    """
    Helper function to create StateChecker Tokens.

    Parameters
    ----------
    n_states : int
        Number of state variables.

    threshold_set : list or list of lists
        A list of constants [t1, t2, ..., tn] for constructing StateChecker (si < tj),
        or a list of lists of constants [[t11, t12, t1n], [t21, t22, ..., t2m], ...].
        In the latter case, the i-th list contains the thresholds for state variable si for 
        constructing StateChecker (si < tij). The sizes of the threshold lists can be different.
    """
    tokens = []
    if isinstance(threshold_set[0], list):
        assert len(threshold_set) == n_states, \
            "If threshold_set is a list of lists, its length must equal n_states."
    else:
        threshold_set = [threshold_set]*n_states

    for i, thresholds in enumerate(threshold_set):
        assert all([U.is_float(t) for t in thresholds]), \
            "threshold_set must contain only real constant numbers."
        tokens.extend([StateChecker(i, t) for t in thresholds])

    return tokens


def create_tokens(n_input_var, function_set, protected, decision_tree_threshold_set=None):
    """
    Helper function to create Tokens.

    Parameters
    ----------
    n_input_var : int
        Number of input variable Tokens.

    function_set : list
        Names of registered Tokens, or floats that will create new Tokens.

    protected : bool
        Whether to use protected versions of registered Tokens.

    decision_tree_threshold_set : list or list of lists
        A list of constants [t1, t2, ..., tn] for constructing nodes (xi < tj) in decision trees,
        or a list of lists of constants [[t11, t12, t1n], [t21, t22, ..., t2m], ...].
        In the latter case, the i-th list contains the thresholds for input variable xi for constructing
        nodes (xi < tij) in decision trees. The sizes of the threshold lists can be different.
    """

    tokens = []

    # Create input variable Tokens
    for i in range(n_input_var):
        token = Token(name="x{}".format(i + 1), arity=0, complexity=1,
                      function=None, input_var=i)
        tokens.append(token)
    print(tokens)

    for op in function_set:

        # Registered Token
        if op in function_map:
            # Overwrite available protected operators
            if protected and not op.startswith("protected_"):
                protected_op = "protected_{}".format(op)
                if protected_op in function_map:
                    op = protected_op

            token = function_map[op]

        # Hard-coded floating-point constant
        elif U.is_float(op):
            token = HardCodedConstant(op)

        # Constant placeholder (to-be-optimized)
        elif op == "const":
            token = PlaceholderConstant()

        elif op == "poly":
            token = Polynomial()

        else:
            print(function_map)
            raise ValueError("Operation {} not recognized.".format(op))

        tokens.append(token)

    if decision_tree_threshold_set is not None and len(decision_tree_threshold_set) > 0:
        state_checkers = create_state_checkers(n_input_var, decision_tree_threshold_set)
        tokens.extend(state_checkers)
        
    return tokens
