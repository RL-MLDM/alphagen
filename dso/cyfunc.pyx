'''
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
'''
# Uncomment the above lines for cProfile

import numpy as np
import array

from dso.library import StateChecker, Polynomial

# Cython specific C imports
cimport numpy as np
from cpython cimport array
cimport cython
from libc.stdlib cimport malloc, free
from cpython.ref cimport PyObject

# Static inits
cdef list apply_stack   = [[None for i in range(25)] for i in range(1024)]
cdef int *stack_count   = <int *> malloc(1024 * sizeof(int))

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function  
def execute(np.ndarray X, int len_traversal, list traversal, int[:] is_input_var):    
            
    """Executes the program according to X.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.
    
    Returns
    -------
    y_hats : array-like, shape = [n_samples]
        The result of executing the program on X.
    """
    #sp              = 0 # allow a dummy first row, requires a none type function with arity of -1
    
    # Init some ints
    cdef int        sp              = -1 # Stack pointer
    cdef int        Xs              = X.shape[0]
    
    # Give cdef hints for object types  
    cdef int        i
    cdef int        n
    cdef int        arity
    cdef np.ndarray intermediate_result
    cdef list       stack_end
    cdef object     stack_end_function
    
    for i in range(len_traversal):
        
        if not is_input_var[i]:
            sp += 1
            # Move this to the front with a memset call
            stack_count[sp]                     = 0
            # Store the reference to stack_count[sp] rather than keep calling
            apply_stack[sp][stack_count[sp]]    = traversal[i]
            stack_end                           = apply_stack[sp]
            # The first element is the function itself
            stack_end_function                  = stack_end[0]
            arity                               = stack_end_function.arity
        else:
            # Not a function, so lazily evaluate later
            stack_count[sp] += 1
            stack_end[stack_count[sp]]          = X[:, traversal[i].input_var]

        # Keep on doing this so long as arity matches up, we can 
        # add in numbers above and complete the arity later.
        while stack_count[sp] == arity:
            # If stack_end_function is a StateChecker (xi < tj), which is associated with
            # the i-th state variable xi and threshold tj, then stack_end_function needs to know
            # the value of xi (through set_state_value) in order to evaluate the returned value.
            # The index i is specified by the attribute state_index of a StateChecker.
            if isinstance(stack_end_function, StateChecker):
                stack_end_function.set_state_value(X[:, stack_end_function.state_index])
            if isinstance(stack_end_function, Polynomial):
                intermediate_result = stack_end_function(X)
            else:
                intermediate_result = stack_end_function(*stack_end[1:(stack_count[sp] + 1)]) # 85% of overhead

            # I think we can get rid of this line, but will require a major rewrite.
            if sp == 0:    
                return intermediate_result
            
            sp -= 1
            # Adjust pointer at the end of the stack
            stack_end                   = apply_stack[sp]
            stack_count[sp] += 1
            stack_end[stack_count[sp]]  = intermediate_result

            # The first element is the function itself
            stack_end_function          = stack_end[0]
            arity                       = stack_end_function.arity
      
    # We should never get here
    assert False, "Function should never get here!"
    return None
