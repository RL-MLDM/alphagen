"""Classes for Token and Library"""

from collections import defaultdict

import numpy as np

import dso.utils as U


class Token():
    """
    An arbitrary token or "building block" of a Program object.

    Attributes
    ----------
    name : str
        Name of token.

    arity : int
        Arity (number of arguments) of token.

    complexity : float
        Complexity of token.

    function : callable
        Function associated with the token; used for exectuable Programs.

    input_var : int or None
        Index of input if this Token is an input variable, otherwise None.

    Methods
    -------
    __call__(input)
        Call the Token's function according to input.
    """

    def __init__(self, function, name, arity, complexity, input_var=None):
        self.function = function
        self.name = name
        self.arity = arity
        self.complexity = complexity
        self.input_var = input_var

        if input_var is not None:
            assert function is None, "Input variables should not have functions."
            assert arity == 0, "Input variables should have arity zero."

    def __call__(self, *args):
        assert self.function is not None, \
            "Token {} is not callable.".format(self.name)

        return self.function(*args)

    def __repr__(self):
        return self.name


class HardCodedConstant(Token):
    """
    A Token with a "value" attribute, whose function returns the value.

    Parameters
    ----------
    value : float
        Value of the constant.
    """

    def __init__(self, value=None, name=None):
        assert value is not None, "Constant is not callable with value None. Must provide a floating point number or string of a float."
        assert U.is_float(value)
        value = np.atleast_1d(np.float32(value))
        self.value = value
        if name is None:
            name = str(self.value[0])
        super().__init__(function=self.function, name=name, arity=0, complexity=1)

    def function(self):
        return self.value


class PlaceholderConstant(Token):
    """
    A Token for placeholder constants that will be optimized with respect to
    the reward function. The function simply returns the "value" attribute.

    Parameters
    ----------
    value : float or None
        Current value of the constant, or None if not yet set.
    """

    def __init__(self, value=None):
        if value is not None:
            value = np.atleast_1d(value)
        self.value = value
        super().__init__(function=self.function, name="const", arity=0, complexity=1)

    def function(self):
        assert self.value is not None, \
            "Constant is not callable with value None."
        return self.value

    def __repr__(self):
        if self.value is None:
            return self.name
        return str(self.value[0])


class Polynomial(Token):
    """
    A Token representing a polynomial of the input variables of the form:
        p(x1, ..., xn) = c1 * x1**e11 * ... * xn**e1n + ... + cm * x1**em1 * ... * xn**emn.
    Note that only the terms with a nonzero coefficient are stored.

    Parameters
    ----------
    exponents : list of tuples of nonnegative integers
        Exponents of the nonzero terms [(e11, ..., e1n), ..., (em1, ..., emn)].
    coef : list of float
        A list of coefficients [c1, c2, ..., cm] corresponding to the terms in exponents.
    """
    def __init__(self, exponents=None, coef=None):
        self.exponents = exponents
        self.coef = coef
        complexity = 1 if coef is None else len(coef)
        super().__init__(self.eval_poly, "poly", arity=0, complexity=complexity)

    @staticmethod
    def eval_monomials(X, monomials_exponents):
        """
        Compute the monomials x1**ej1 * ... * xn**ejn for each data point [x1, ..., xn] in X.
        Parameters
        ----------
        X : ndarray
            A dataset of shape (number of points, number of variables).
        monomials_exponents : list of tuples of nonnegative integers
            Exponents of the monomials to be computed: [(e11, ..., e1n), ..., (em1, ..., emn)].
        """
        monomials = np.ones(shape=(X.shape[0], len(monomials_exponents)))
        for basis_count, exponents in enumerate(monomials_exponents):
            for i in range(len(exponents)):
                if exponents[i] != 0:
                    monomials[:, basis_count] *= X[:, i] ** exponents[i]
        return monomials

    def eval_poly(self, X):
        assert self.exponents is not None and self.coef is not None
        if len(self.coef) == 0:
            return np.zeros(shape=(X.shape[0], 1))
        return np.dot(self.eval_monomials(X, self.exponents), self.coef)

    def __repr__(self):
        if self.exponents is None and self.coef is None:
            return self.name

        if len(self.coef) == 0:
            return "0.0"

        assert len(self.exponents) == self.coef.shape[0]
        names = ["-" if self.coef[0] < 0 else ""]
        for basis_count, exponents in enumerate(self.exponents):
            basis_name = [str(format(np.abs(self.coef[basis_count]), '.6'))]
            for i in range(len(exponents)):
                if exponents[i] == 1:
                    basis_name.append("x{}".format(i + 1))
                elif exponents[i] > 1:
                    basis_name.append("x{}**{}".format(i + 1, int(exponents[i])))
            names.append("*".join(basis_name))
            if basis_count < len(self.coef) - 1:
                names.append("-" if self.coef[basis_count + 1] < 0 else "+")
        return "".join(names)

    def to_str_tokens(self):
        """
        Return a list of tokens of add, mul, inputs, and constants that 
        is equivalent to this poly token.
        """
        if self.exponents is None and self.coef is None:
            return []

        assert len(self.exponents) == self.coef.shape[0]
        out = [] if len(self.coef) == 0 else ["add"]*(len(self.coef)-1)
        for n, exponents in enumerate(self.exponents):
            out.extend(["mul"]*np.count_nonzero(exponents))
            for i in range(len(exponents)):
                if exponents[i] >= 1:
                    out.extend(["mul"]*(exponents[i]-1))
                    out.extend(["x{}".format(i + 1)]*exponents[i])
            out.append(self.coef[n])
        return out


class StateChecker(Token):
    """
    A Token for making decisions in decision trees. Given the i-th state
    variable xi and a threshold tj, the associated function is:
        xi_<_tj(f, g) = f if xi < tj else g.
    Here, the index i is indicated by state_index in __init__.

    Note that:
        1. The arity of StateChecker is 2, so subroutines designed for Tokens
           of arity <= 2 can also be applied to StateChecker.
        2. If StateCheckers are included in the library, it is recommended to
           turn on the StateCheckerConstraint, which prevents sampling of
           Tokens that lead to degenerate situations like "checking if xi < 6
           when we know xi < 3".
        3. When a StateChecker is initialized, state_value (the value of xi)
           is unknown. In order to evaluate the return value of the associated
           function, state_value needs to be set before the evaluation
           (typically during the execution of a Program). See cyfunc.pyx for
           an example usage.

    Parameters
    ----------
    state_index : int
        Index of the state variable associated with the token.

    threshold : float
        Value to which the state variable is compared when making decisions.
    """

    def __init__(self, state_index, threshold):
        assert threshold is not None, \
                "StateChecker requires a float value for threshold."
        self.state_index = state_index
        self.state_value = None
        self.threshold = threshold

        name = "x{} < {}".format(state_index + 1, self.threshold)
        super().__init__(function=self.function, name=name, arity=2, complexity=1)

    def set_state_value(self, state_value):
        self.state_value = state_value

    def function(self, value_if_true, value_if_false):
        assert self.state_value is not None, "StateChecker.state_value has not been set."
        return np.where(np.less(self.state_value, self.threshold), value_if_true, value_if_false)


class DiscreteAction(HardCodedConstant):
    """
    This class is intended to be used for learning decision tree policies when
    the env of the control problem has a Discrete action space.
    Discrete action a_i corresponds to constant value i-1, i = 1, 2, 3, ...
    """
    def __init__(self, value):
        assert isinstance(value, int) and value >= 0
        super().__init__(value, "a_{}".format(value+1))


class MultiDiscreteAction(Token):
    """
    This class is intended to be used for learning decision tree policies when
    the env of the control problem has a MultiDiscrete action space.

    Tokens in this class are ai_j (unary) and STOP (terminal). When executed:
        - ai_j returns an array that sets the (i-1)-th value of input to j-1.
        - STOP returns an array of default actions of all dimensions.
    For example, supposed the default actions are set to be [2, 1, 0].
    The traversal [a1_2, STOP] corresponds to the constant action [1, 1, 0],
    while the traversal [a1_3, a3_3, a2_1, STOP] corresponds to [2, 0, 2].
    """
    n_dims = None # total number of action dimensions
    def __init__(self, value, action_dim=None):
        """
        Parameters
        ----------
        value : int or list of int
            If action_dim is not None, the token corresponds to the value-th
            discrete action in action dimension action_dim.
            Otherwise, it is a list of integers specifying the default discrete
            actions of all action dimensions.

        action_dim : int or None
            Action dimension. If None, the STOP token will be constructed.
        """
        self.value = value
        self.action_dim = action_dim

        if action_dim is None:
            assert isinstance(self.value, list)
            MultiDiscreteAction.n_dims = len(self.value)
            self.value = np.array(self.value)
            name = "STOP"
            super().__init__(function=self.apply_action, name=name, arity=0, complexity=1)
        else:
            assert isinstance(value, int) and value >= 0
            name = "a{}_{}".format(action_dim+1, value+1)
            super().__init__(function=self.apply_action, name=name, arity=1, complexity=1)
            
    def apply_action(self, *args):
        if self.action_dim is None:
            return np.array([self.value.copy()])
        else:
            args[0][0, self.action_dim] = self.value
            return args[0]


class Library():
    """
    Library of Tokens. We use a list of Tokens (instead of set or dict) since
    we so often index by integers given by the Controller.

    Attributes
    ----------
    tokens : list of Token
        List of available Tokens in the library.

    names : list of str
        Names corresponding to Tokens in the library.

    arities : list of int
        Arities corresponding to Tokens in the library.
    """

    def __init__(self, tokens):

        self.tokens = tokens
        self.L = len(tokens)
        self.names = [t.name for t in tokens]
        self.arities = np.array([t.arity for t in tokens], dtype=np.int32)

        self.input_tokens = np.array(
            [i for i, t in enumerate(self.tokens) if t.input_var is not None],
            dtype=np.int32)

        self.state_checker_tokens = np.array(
            [i for i, t in enumerate(self.tokens) if isinstance(t, StateChecker)],
            dtype=np.int32)

        self.multi_discrete_tokens = np.array(
            [i for i, t in enumerate(self.tokens) if isinstance(t, MultiDiscreteAction)],
            dtype=np.int32)            

        def get_tokens_of_arity(arity):
            _tokens = [i for i in range(self.L) if self.arities[i] == arity]
            return np.array(_tokens, dtype=np.int32)

        self.tokens_of_arity = defaultdict(lambda : np.array([], dtype=np.int32))
        for arity in self.arities:
            self.tokens_of_arity[arity] = get_tokens_of_arity(arity)
        self.terminal_tokens = self.tokens_of_arity[0]
        self.unary_tokens = self.tokens_of_arity[1]
        self.binary_tokens = self.tokens_of_arity[2]

        try:
            self.const_token = self.names.index("const")
        except ValueError:
            self.const_token = None
        try:
            self.poly_token = self.names.index("poly")
        except ValueError:
            self.poly_token = None
        self.parent_adjust = np.full_like(self.arities, -1)
        count = 0
        for i in range(len(self.arities)):
            if self.arities[i] > 0:
                self.parent_adjust[i] = count
                count += 1

        trig_names = ["sin", "cos", "tan", "csc", "sec", "cot"]
        trig_names += ["arc" + name for name in trig_names]

        self.float_tokens = np.array(
            [i for i, t in enumerate(self.tokens) if t.arity == 0 and t.input_var is None],
            dtype=np.int32)
        self.trig_tokens = np.array(
            [i for i, t in enumerate(self.tokens) if t.name in trig_names],
            dtype=np.int32)

        inverse_tokens = {
            "inv" : "inv",
            "neg" : "neg",
            "exp" : "log",
            "log" : "exp",
            "sqrt" : "n2",
            "n2" : "sqrt"
        }
        token_from_name = {t.name : i for i, t in enumerate(self.tokens)}
        self.inverse_tokens = {token_from_name[k] : token_from_name[v] for k, v in inverse_tokens.items() if k in token_from_name and v in token_from_name}

        self.n_action_inputs = self.L + 1 # Library tokens + empty token
        self.n_parent_inputs = self.L + 1 - len(self.terminal_tokens) # Parent sub-lib tokens + empty token
        self.n_sibling_inputs = self.L + 1 # Library tokens + empty token
        self.n_input_tokens = len(self.input_tokens)
        self.EMPTY_ACTION = self.n_action_inputs - 1
        self.EMPTY_PARENT = self.n_parent_inputs - 1
        self.EMPTY_SIBLING = self.n_sibling_inputs - 1

    def __getitem__(self, val):
        """Shortcut to get Token by name or index."""

        if isinstance(val, str):
            try:
                i = self.names.index(val)
            except ValueError:
                raise TokenNotFoundError("Token {} does not exist.".format(val))
        elif isinstance(val, (int, np.integer)):
            i = val
        else:
            raise TokenNotFoundError("Library must be indexed by str or int, not {}.".format(type(val)))

        try:
            token = self.tokens[i]
        except IndexError:
            raise TokenNotFoundError("Token index {} does not exist".format(i))
        return token

    def tokenize(self, inputs):
        """Convert inputs to list of Tokens."""

        # TBD non-list should return non-list

        if isinstance(inputs, str):
            inputs = inputs.split(',')
        elif not isinstance(inputs, list) and not isinstance(inputs, np.ndarray): # TBD FIX HACK
            inputs = [inputs]
        tokens = [input_ if isinstance(input_, Token) else self[input_] for input_ in inputs]
        return tokens

    def actionize(self, inputs):
        """Convert inputs to array of 'actions', i.e. ints corresponding to
        Tokens in the Library."""

        tokens = self.tokenize(inputs)
        actions = np.array([self.tokens.index(t) for t in tokens],
                           dtype=np.int32)
        return actions


class TokenNotFoundError(Exception):
    pass
