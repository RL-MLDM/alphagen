"""Class for symbolic expression object or program."""

import array
import os
import warnings
from textwrap import indent

import numpy as np
from dso.library import Token, PlaceholderConstant, Polynomial
from dso.const import make_const_optimizer
from dso.utils import cached_property
import dso.utils as U



def _finish_tokens(tokens):

    """
    Complete a possibly unfinished string of tokens.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal.

    Returns
    _______
    tokens : list of ints
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    """

    if Program.task.task_type == "binding":
        return tokens

    arities = np.array([Program.library.arities[t] for t in tokens])
    # Number of dangling nodes, returns the cumsum up to each point
    # Note that terminal nodes are -1 while functions will be >= 0 since arities - 1
    dangling = 1 + np.cumsum(arities - 1)

    if -1 in (dangling - 1):
        # chop off tokens once the cumsum reaches 0, this is the last valid point in the tokens
        expr_length = 1 + np.argmax((dangling - 1) == -1)
        tokens = tokens[:expr_length]
    else:
        # Extend with valid variables until string is valid
        # NOTE: This only appends onto the end of a set of tokens, even in the multi-object case!
        if Program.task.task_type != 'binding':
            tokens = np.append(tokens, np.random.choice(Program.library.input_tokens, size=dangling[-1]))

    return tokens


def from_str_tokens(str_tokens, skip_cache=False):
    """
    Memoized function to generate a Program from a list of str and/or float.
    See from_tokens() for details.

    Parameters
    ----------
    str_tokens : str | list of (str | float)
        Either a comma-separated string of tokens and/or floats, or a list of
        str and/or floats.

    skip_cache : bool
        See from_tokens().

    Returns
    -------
    program : Program
        See from_tokens().
    """

    # Convert str to list of str
    if isinstance(str_tokens, str):
        str_tokens = str_tokens.split(",")

    # Convert list of str|float to list of tokens
    if isinstance(str_tokens, list):
        traversal = []
        constants = []
        for s in str_tokens:
            if s in Program.library.names:
                t = Program.library.names.index(s.lower())
            elif U.is_float(s):
                assert "const" not in str_tokens, "Currently does not support both placeholder and hard-coded constants."
                t = Program.library.const_token
                constants.append(float(s))
            else:
                raise ValueError("Did not recognize token {}.".format(s))
            traversal.append(t)
        traversal = np.array(traversal, dtype=np.int32)
    else:
        raise ValueError("Input must be list or string.")

    # Generate base Program (with "const" for constants)
    p = from_tokens(traversal, skip_cache=skip_cache)

    # Replace any constants
    p.set_constants(constants)

    return p


def from_tokens(tokens, skip_cache=False, on_policy=True, finish_tokens=True):

    """
    Memoized function to generate a Program from a list of tokens.

    Since some tokens are nonfunctional, this first computes the corresponding
    traversal. If that traversal exists in the cache, the corresponding Program
    is returned. Otherwise, a new Program is returned.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    skip_cache : bool
        Whether to bypass the cache when creating the program (used for
        previously learned symbolic actions in DSP).
        
    finish_tokens: bool
        Do we need to finish this token. There are instances where we have
        already done this. Most likely you will want this to be True. 

    Returns
    _______
    program : Program
        The Program corresponding to the tokens, either pulled from memoization
        or generated from scratch.
    """

    '''
        Truncate expressions that complete early; extend ones that don't complete
    '''
  
    if finish_tokens:
        tokens = _finish_tokens(tokens)

    # For stochastic Tasks, there is no cache; always generate a new Program.
    # For deterministic Programs, if the Program is in the cache, return it;
    # otherwise, create a new one and add it to the cache.
    if skip_cache or Program.task.stochastic:
        p = Program(tokens, on_policy=on_policy)
    else:
        key = tokens.tostring()
        try:
            p = Program.cache[key]
            if on_policy:
                p.on_policy_count += 1
            else:
                p.off_policy_count += 1
        except KeyError:
            p = Program(tokens, on_policy=on_policy)
            Program.cache[key] = p

    return p


class Program(object):
    """
    The executable program representing the symbolic expression.

    The program comprises unary/binary operators, constant placeholders
    (to-be-optimized), input variables, and hard-coded constants.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. "Dangling"
        programs are completed with repeated "x1" until the expression
        completes.

    Attributes
    ----------
    traversal : list
        List of operators (type: Function) and terminals (type: int, float, or
        str ("const")) encoding the pre-order traversal of the expression tree.

    tokens : np.ndarry (dtype: int)
        Array of integers whose values correspond to indices

    const_pos : list of int
        A list of indicies of constant placeholders along the traversal.

    float_pos : list of float
        A list of indices of constants placeholders or floating-point constants
        along the traversal.

    poly_pos : int
        Index of poly token in the traversal if it has one.

    sympy_expr : str
        The (lazily calculated) SymPy expression corresponding to the program.
        Used for pretty printing _only_.

    complexity : float
        The (lazily calcualted) complexity of the program.

    r : float
        The (lazily calculated) reward of the program.

    count : int
        The number of times this Program has been sampled.

    str : str
        String representation of tokens. Useful as unique identifier.
    """

    # Static variables
    task = None             # Task
    library = None          # Library
    const_optimizer = None  # Function to optimize constants
    cache = {}

    # Cython-related static variables
    have_cython = None      # Do we have cython installed
    execute = None          # Link to execute. Either cython or python

    def __init__(self, tokens=None, on_policy=True):
        """
        Builds the Program from a list of of integers corresponding to Tokens.
        """
        
        # Can be empty if we are unpickling 
        if tokens is not None:
            self._init(tokens, on_policy)
            
    def _init(self, tokens, on_policy=True):

        self.traversal = [Program.library[t] for t in tokens]
        self.const_pos = [i for i, t in enumerate(self.traversal) if isinstance(t, PlaceholderConstant)]
        poly_pos = [i for i, t in enumerate(self.traversal) if isinstance(t, Polynomial)]
        assert len(poly_pos) <= 1, "A program cannot contain more than one 'poly' token"
        self.poly_pos = poly_pos[0] if len(poly_pos) > 0 else None
        self.len_traversal = len(self.traversal)

        if self.have_cython and self.len_traversal > 1:
            self.is_input_var = array.array('i', [t.input_var is not None for t in self.traversal])

        self.invalid = False
        self.str = tokens.tostring()
        self.tokens = tokens

        self.on_policy_count = 1 if on_policy else 0
        self.off_policy_count = 0 if on_policy else 1
        self.originally_on_policy = on_policy # Note if a program was created on policy

    def execute(self, X):
        """
        Execute program on input X.

        Parameters
        ==========

        X : np.array
            Input to execute the Program over.

        Returns
        =======

        result : np.array or list of np.array
            In a single-object Program, returns just an array. In a multi-object Program, returns a list of arrays.
        """
        if not Program.protected:
            result, self.invalid, self.error_node, self.error_type = Program.execute_function(self.traversal, X)
        else:
            result = Program.execute_function(self.traversal, X)
        return result

    def optimize(self):
        """
        Optimizes PlaceholderConstant tokens against the reward function. The
        optimized values are stored in the traversal.
        """

        # TBD: Should use np.float32

        if len(self.const_pos) == 0:
            return

        # Define the objective function: negative reward
        def f(consts):
            self.set_constants(consts)
            r = self.task.reward_function(self, optimizing=True)
            obj = -r # Constant optimizer minimizes the objective function

            # Need to reset to False so that a single invalid call during
            # constant optimization doesn't render the whole Program invalid.
            self.invalid = False

            return obj

        # Do the optimization
        x0 = np.ones(len(self.const_pos)) # Initial guess
        optimized_constants = Program.const_optimizer(f, x0)

        # Set the optimized constants
        self.set_constants(optimized_constants)

    def get_constants(self):
        """Returns the values of a Program's constants."""

        return [t.value for t in self.traversal if isinstance(t, PlaceholderConstant)]

    def set_constants(self, consts):
        """Sets the program's constants to the given values"""

        for i, const in enumerate(consts):
            assert U.is_float, "Input to program constants must be of a floating point type"
            # Create a new instance of PlaceholderConstant instead of changing
            # the "values" attribute, otherwise all Programs will have the same
            # instance and just overwrite each other's value.
            self.traversal[self.const_pos[i]] = PlaceholderConstant(const)

    def get_poly(self):
        """Returns a Program's Polynomial token if it has one."""

        return None if self.poly_pos is None else self.traversal[self.poly_pos]

    def set_poly(self, poly_token):
        """Sets the program's Polynomial token to the given token"""

        if self.poly_pos is not None:
            self.traversal[self.poly_pos] = poly_token


    @classmethod
    def clear_cache(cls):
        """Clears the class' cache"""

        cls.cache = {}


    @classmethod
    def set_task(cls, task):
        """Sets the class' Task"""

        Program.task = task
        Program.library = task.library


    @classmethod
    def set_const_optimizer(cls, name, **kwargs):
        """Sets the class' constant optimizer"""

        const_optimizer = make_const_optimizer(name, **kwargs)
        Program.const_optimizer = const_optimizer

    @classmethod
    def set_complexity(cls, name):
        """Sets the class' complexity function"""

        all_functions = {
            # No complexity
            None : lambda p : 0.0,

            # Length of sequence
            "length" : lambda p : len(p.traversal),

            # Sum of token-wise complexities
            "token" : lambda p : sum([t.complexity for t in p.traversal]),

            # Binding complexity: % of mutations relative to master seq
            "mutations" : lambda p : Program.task.compute_mutational_distance(p)
        }

        assert name in all_functions, "Unrecognized complexity function name."

        Program.complexity_function = lambda p : all_functions[name](p)

    @classmethod
    def set_execute(cls, protected):
        """Sets which execute method to use"""

        # Check if cython_execute can be imported; if not, fall back to python_execute
        from dso.execute import python_execute
        execute_function = python_execute
        Program.have_cython = False

        if protected:
            Program.protected = True
            Program.execute_function = execute_function
        else:
            Program.protected = False
            class InvalidLog():
                """Log class to catch and record numpy warning messages"""

                def __init__(self):
                    self.error_type = None # One of ['divide', 'overflow', 'underflow', 'invalid']
                    self.error_node = None # E.g. 'exp', 'log', 'true_divide'
                    self.new_entry = False # Flag for whether a warning has been encountered during a call to Program.execute()

                def write(self, message):
                    """This is called by numpy when encountering a warning"""

                    if not self.new_entry: # Only record the first warning encounter
                        message = message.strip().split(' ')
                        self.error_type = message[1]
                        self.error_node = message[-1]
                    self.new_entry = True

                def update(self):
                    """If a floating-point error was encountered, set Program.invalid
                    to True and record the error type and error node."""

                    if self.new_entry:
                        self.new_entry = False
                        return True, self.error_type, self.error_node
                    else:
                        return False, None, None


            invalid_log = InvalidLog()
            np.seterrcall(invalid_log) # Tells numpy to call InvalidLog.write() when encountering a warning

            # Define closure for execute function
            def unsafe_execute(traversal, X):
                """This is a wrapper for execute_function. If a floating-point error
                would be hit, a warning is logged instead, p.invalid is set to True,
                and the appropriate nan/inf value is returned. It's up to the task's
                reward function to decide how to handle nans/infs."""

                with np.errstate(all='log'):
                    y = execute_function(traversal, X)
                    invalid, error_node, error_type = invalid_log.update()
                    return y, invalid, error_node, error_type

            Program.execute_function = unsafe_execute
                
    @cached_property
    def r(self):
        """Evaluates and returns the reward of the program"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Optimize any PlaceholderConstants
            self.optimize()

            # Return final reward after optimizing
            return self.task.reward_function(self)

    @cached_property
    def complexity(self):
        """Evaluates and returns the complexity of the program"""

        return Program.complexity_function(self)

    @cached_property
    def evaluate(self):
        """Evaluates and returns the evaluation metrics of the program."""

        # Program must be optimized before computing evaluate
        if "r" not in self.__dict__:
            print("WARNING: Evaluating Program before computing its reward." \
                  "Program will be optimized first.")
            self.optimize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return self.task.evaluate(self)

    @cached_property
    def sympy_expr(self):
        """
        Returns the attribute self.sympy_expr.

        This is actually a bit complicated because we have to go: traversal -->
        tree --> serialized tree --> SymPy expression
        """

        tree = self.traversal.copy()
        tree = build_tree(tree)
        tree = convert_to_sympy(tree)
        try:
            expr = U.parse_expr(tree.__repr__()) # SymPy expression
        except:
            expr = tree.__repr__()
        return expr

    def pretty(self):
        """Returns pretty printed string of the program"""

        if self.task.task_type != "binding":
            return U.pretty(self.sympy_expr)
        else:
            return None

    def print_stats(self):
        """Prints the statistics of the program

            We will print the most honest reward possible when using validation.
        """

        print("\tReward: {}".format(self.r))
        print("\tCount Off-policy: {}".format(self.off_policy_count))
        print("\tCount On-policy: {}".format(self.on_policy_count))
        print("\tOriginally on Policy: {}".format(self.originally_on_policy))
        print("\tInvalid: {}".format(self.invalid))
        print("\tTraversal: {}".format(self))
        if self.task.task_type != 'binding':
            print("\tExpression:") 
            print("{}\n".format(indent(self.pretty(), '\t  ')))

    def __repr__(self):
        """Prints the program's traversal"""
        return ','.join([repr(t) for t in self.traversal])


###############################################################################
# Everything below this line is currently only being used for pretty printing #
###############################################################################


# Possible library elements that sympy capitalizes
capital = ["add", "mul", "pow"]


class Node(object):
    """Basic tree class supporting printing"""

    def __init__(self, val):
        self.val = val
        self.children = []

    def __repr__(self):
        children_repr = ",".join(repr(child) for child in self.children)
        if len(self.children) == 0:
            return self.val # Avoids unnecessary parantheses, e.g. x1()
        return "{}({})".format(self.val, children_repr)


def build_tree(traversal):
    """Recursively builds tree from pre-order traversal"""

    op = traversal.pop(0)
    n_children = op.arity
    val = repr(op)
    if val in capital:
        val = val.capitalize()

    node = Node(val)

    for _ in range(n_children):
        node.children.append(build_tree(traversal))

    return node


def convert_to_sympy(node):
    """Adjusts trees to only use node values supported by sympy"""

    if node.val == "div":
        node.val = "Mul"
        new_right = Node("Pow")
        new_right.children.append(node.children[1])
        new_right.children.append(Node("-1"))
        node.children[1] = new_right

    elif node.val == "sub":
        node.val = "Add"
        new_right = Node("Mul")
        new_right.children.append(node.children[1])
        new_right.children.append(Node("-1"))
        node.children[1] = new_right

    elif node.val == "inv":
        node.val = Node("Pow")
        node.children.append(Node("-1"))

    elif node.val == "neg":
        node.val = Node("Mul")
        node.children.append(Node("-1"))

    elif node.val == "n2":
        node.val = "Pow"
        node.children.append(Node("2"))

    elif node.val == "n3":
        node.val = "Pow"
        node.children.append(Node("3"))

    elif node.val == "n4":
        node.val = "Pow"
        node.children.append(Node("4"))

    for child in node.children:
        convert_to_sympy(child)

    return node
