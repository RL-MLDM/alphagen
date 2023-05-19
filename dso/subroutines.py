"""Numba-compiled subroutines used for deep symbolic optimization."""

from numba import jit, prange
import numpy as np


@jit(nopython=True, parallel=True)
def parents_siblings(tokens, arities, parent_adjust, empty_parent, empty_sibling):
    """
    Given a batch of action sequences, computes and returns the parents and
    siblings of the next element of the sequence.

    The batch has shape (N, L), where N is the number of sequences (i.e. batch
    size) and L is the length of each sequence. In some cases, expressions may
    already be complete; in these cases, this function sees the start of a new
    expression, even though the return value for these elements won't matter
    because their gradients will be zero because of sequence_length.

    Parameters
    __________

    tokens : np.ndarray, shape=(N, L), dtype=np.int32
        Batch of action sequences. Values correspond to library indices.

    arities : np.ndarray, dtype=np.int32
        Array of arities corresponding to library indices.

    parent_adjust : np.ndarray, dtype=np.int32
        Array of parent sub-library index corresponding to library indices.

    empty_parent : int
        Integer value for an empty parent token. This is initially computed in controller.py.

    empty_sibling : int
        Integer value for an empty sibling token. This is intially computed in controller.py

    Returns
    _______

    adj_parents : np.ndarray, shape=(N,), dtype=np.int32
        Adjusted parents of the next element of each action sequence.

    siblings : np.ndarray, shape=(N,), dtype=np.int32
        Siblings of the next element of each action sequence.

    """
    N, L = tokens.shape

    adj_parents = np.full(shape=(N,), fill_value=empty_parent, dtype=np.int32)
    siblings = np.full(shape=(N,), fill_value=empty_sibling, dtype=np.int32)
    # Parallelized loop over action sequences
    for r in prange(N):
        arity = arities[tokens[r, -1]]
        if arity > 0: # Parent is the previous element; no sibling
            adj_parents[r] = parent_adjust[tokens[r, -1]]
            continue
        dangling = 0
        # Loop over elements in an action sequence
        for c in range(L):
            arity = arities[tokens[r, L - c - 1]]
            dangling += arity - 1
            if dangling == 0: # Parent is L-c-1, sibling is the next
                adj_parents[r] = parent_adjust[tokens[r, L - c - 1]]
                siblings[r] = tokens[r, L - c]
                break
    return adj_parents, siblings


# TBD: Refactor to compute hierarchical obs
@jit(nopython=True, parallel=False)
def jit_parents_siblings_at_once(tokens, arities, parent_adjust):
    """
    Given a batch of action sequences, computes and returns the parents and
    siblings over the entire sequence at once.
    
    This version will give all parents and siblings at once over the full
    and complete set of tokens. This is useful for Deap because it generates
    each sequence in one go rather than one token at a time. 

    The batch has shape (N, L), where N is the number of sequences (i.e. batch
    size) and L is the length of each sequence. In some cases, expressions may
    already be complete; in these cases, this function sees the start of a new
    expression, even though the return value for these elements won't matter
    because their gradients will be zero because of sequence_length.
    
    >>> This has been tested and gives the same answer as the regular parent_sibling class for
        DEAP functions. 

    Parameters
    __________

    tokens : np.ndarray, shape=(N, L), dtype=np.int32
        Batch of action sequences. Values correspond to library indices.

    arities : np.ndarray, dtype=np.int32
        Array of arities corresponding to library indices.

    parent_adjust : np.ndarray, dtype=np.int32
        Array of parent sub-library index corresponding to library indices.

    Returns
    _______

    adj_parents : np.ndarray, shape=(N, L), dtype=np.int32
        Adjusted parents of the next element of each action sequence.

    siblings : np.ndarray, shape=(N, L), dtype=np.int32
        Siblings of the next element of each action sequence.
        


    """
    N, L = tokens.shape

    empty_parent    = np.max(parent_adjust) + 1 # Empty token is after all non-empty tokens
    empty_sibling   = len(arities) # Empty token is after all non-empty tokens
    adj_parents     = np.full(shape=(N,L), fill_value=empty_parent, dtype=np.int32)
    siblings        = np.full(shape=(N,L), fill_value=empty_sibling, dtype=np.int32)
    
    # Parallelization is slower here ...
        
    # We loop over actions since frequently, N is 1 when used with Deap. 
    for b in range(1, L):
        for r in range(N):
            # This part is optimal
            arity = arities[tokens[r, b - 1]]            
            if arity > 0: # Parent is the previous element; no sibling
                adj_parents[r, b]   = parent_adjust[tokens[r, b - 1]]
                continue
            
            # This part may not be optimal here, but is fast enough for now
            dangling = 0
            # Loop over elements in an action sequence GOING BACKWARDS
            for c in range(b):
                arity = arities[tokens[r, b - c - 1]]
                dangling += arity - 1
                
                # Most recent non-dangling action
                if dangling == 0: 
                    # Parent is b-c-1, sibling is the next
                    adj_parents[r, b]   = parent_adjust[tokens[r, b - c - 1]]
                    siblings[r, b]      = tokens[r, b - c]
                    break
               
    return adj_parents, siblings


@jit(nopython=True, parallel=True)
def ancestors(actions, arities, ancestor_tokens):
    """
    Given a batch of action sequences, determines whether the next element of
    the sequence has an ancestor in ancestor_tokens.

    The batch has shape (N, L), where N is the number of sequences (i.e. batch
    size) and L is the length of each sequence. In some cases, expressions may
    already be complete; in these cases, this function sees the start of a new
    expression, even though the return value for these elements won't matter
    because their gradients will be zero because of sequence_length.

    Parameters
    __________

    actions : np.ndarray, shape=(N, L), dtype=np.int32
        Batch of action sequences. Values correspond to library indices.

    arities : np.ndarray, dtype=np.int32
        Array of arities corresponding to library indices.

    ancestor_tokens : np.ndarray, dtype=np.int32
        Array of ancestor library indices to check.

    Returns
    _______

    mask : np.ndarray, shape=(N,), dtype=np.bool_
        Mask of whether the next element of each sequence has an ancestor in
        ancestor_tokens.
    """

    for token in ancestor_tokens:
        assert arities[token] == 1, "subroutine 'ancestors' may not work" \
            "properly for non-unary ancestor_tokens"

    N, L = actions.shape
    mask = np.zeros(shape=(N,), dtype=np.bool_)
    # Parallelized loop over action sequences
    for r in prange(N):
        dangling = 0
        threshold = None # If None, current branch does not have trig ancestor
        for c in range(L):
            arity = arities[actions[r, c]]
            dangling += arity - 1
            # Turn "on" if a trig function is found
            # Remain "on" until branch completes
            if threshold is None:
                for trig_token in ancestor_tokens:
                    if actions[r, c] == trig_token:
                        threshold = dangling - 1
                        break
            # Turn "off" once the branch completes
            else:
                if dangling == threshold:
                    threshold = None
        # If the sequences ended "on", then there is a trig ancestor
        if threshold is not None:
            mask[r] = True
    return mask


@jit(nopython=True, parallel=False)
def jit_check_constraint_violation(actions, actions_tokens, other, other_tokens):
    r"""
    Given an action sequence, another type of sequences such as siblings 
    or children and constraint tokens, this will return a bool which tells if
    the constraint was violated. 

    The batch has shape (1, L), L is the length of the sequence.
    
    This does the same thing as:
    
        np.any(np.logical_and(np.isin(actions, actions_tokens), np.isin(other, other_tokens)))
    
    but is much faster because it can quit when a single constraint is violated. 
    
    >>> This has been tested against the old inverse token constraint and gives the same answer.

    Parameters
    __________

    actions : np.ndarray, shape=(1, L), dtype=np.int32
        Batch of action sequences. Values correspond to library indices.
        
    actions_tokens : np.ndarray, dtype=np.int32
        Array of constraint tokens to match action against.

    other : np.ndarray, shape=(1, L), dtype=np.int32
        Batch of other sequences. Values correspond to library indices.
        
    other_tokens : np.ndarray, dtype=np.int32
        Array of constraint tokens to match other against.

    Returns
    _______

    bool : Was the constraint violated. 
    
    """
    # Is this token item A found in the list of tokens in B?
    def a_in_b(a, B_tokens, B):
        for b in range(B):
            if a == B_tokens[b]:
                return True
        return False
     
    _,L     = actions.shape
    A       = actions_tokens.shape[0]
    O       = other_tokens.shape[0]
    
    # For each action:
    for l in range(L):
        # Check if this token matches a constraint token
        # And check if the other also matches one of its constraints
        if a_in_b(actions[0,l], actions_tokens, A) and a_in_b(other[0,l], other_tokens, O):
            return True
                    
    return False


@jit(nopython=True, parallel=False)
def jit_check_constraint_violation_uchild(actions, parent, sibling, actions_tokens, 
                                          adj_unary_effectors, adj_effectors):
    r"""
    Given an action sequence, another type of sequences such as siblings 
    or children and constraint tokens, this will return a bool which tells if
    the constraint was violated. 

    The batch has shape (1, L), L is the length of the sequence.
    
    This does the same thing as:
    
        for i, a in enumerate(actions):
            if (parent[i] in adj_unary_effectors) or (sibling[i] in self.targets and parent[i] in adj_effectors)
                if a in self.targets:
                    return True
    
    but is much faster because it can quit when a single constraint is violated. 
    
 
    Parameters
    __________

    actions : np.ndarray, shape=(1, L), dtype=np.int32
        Batch of action sequences. Values correspond to library indices.
        
    parent  : np.ndarray, shape=(1, L), dtype=np.int32
        Batch of parent sequences. Values correspond to library indices.
        
    sibling : np.ndarray, shape=(1, L), dtype=np.int32
        Batch of sibling sequences. Values correspond to library indices.
        
    actions_tokens : np.ndarray, dtype=np.int32
        Array of constraint tokens to match action against.

    adj_unary_effectorss : np.ndarray, dtype=np.int32
        Array of constraint tokens to match action against.

    adj_effectors : np.ndarray, dtype=np.int32
        Array of constraint tokens to match action against.
    
    Returns
    _______

    bool : Was the constraint violated. 
    
    """
    # Is this token item A found in the list of tokens in B?
    def a_in_b(a, B_tokens, B):
        for b in range(B):
            if a == B_tokens[b]:
                return True
        return False
     
    _,L     = actions.shape
    A       = actions_tokens.shape[0]
    U       = adj_unary_effectors.shape[0]
    E       = adj_effectors.shape[0]
    
    # For each action:
    for l in range(L):
        
        # Is this the right action?
        if a_in_b(actions[0,l], actions_tokens, A):
            # CASE 1: parent is a unary effector
            if a_in_b(parent[0,l], adj_unary_effectors, U):
                return True
            
            # CASE 2: sibling is a target and parent is an effector
            if a_in_b(sibling[0,l], actions_tokens, A) and a_in_b(parent[0,l], adj_effectors, E):
                return True
            
    return False


@jit(nopython=True, parallel=False)
def jit_check_constraint_violation_descendant_no_target_tokens(\
        actions, effector_tokens, binary_tokens, unary_tokens):

    r"""
    Given an action sequence, another type of sequences such as siblings 
    or children and constraint tokens, this will return a bool which tells if
    the constraint was violated. 

    The batch has shape (1, L), L is the length of the sequence.
    
    This can be used (for instance) to check for trig constraints. 
    
    This does the same thing as:
    
        descendant = False # True when current node is a descendant of operator
        
        for a in actions:
            if a in self.targets:
                if descendant:
                    return True
                descendant = True
                dangling   = 1
            elif descendant:
                if a in library.binary_tokens:      
                    dangling += 1
                elif a not in library.unary_tokens: 
                    dangling -= 1
                if dangling == 0:
                    descendant = False
                    
        return False
        
    >>> This has been tested against the old Trig constraint and gives the same answer
    
    Parameters
    __________

    actions : np.ndarray, shape=(1, L), dtype=np.int32
        Batch of action sequences. Values correspond to library indices.
        
    effector_tokens : np.ndarray, dtype=np.int32
        Array of constraint tokens to match action against.

    binrary_tokens : np.ndarray, dtype=np.int32
        Array of binary function tokens in the current library. 
        
    uniary_tokens : np.ndarray, dtype=np.int32
        Array of unary function tokens in the current library. 

    Returns
    _______

    bool : Was the constraint violated. 
    
    """

    # Is this token item A found in the list of tokens in B?
    def a_in_b(a, B_tokens, B):
        for b in range(B):
            if a == B_tokens[b]:
                return True
        return False
    
    # Is this token item A NOT found in the list of tokens in B?                        
    def a_not_in_b(a, B_tokens, B):
        for b in range(B):
            if a == B_tokens[b]:
                return False
        return True
            
    _,L     = actions.shape
    E       = effector_tokens.shape[0]
    B       = binary_tokens.shape[0]
    U       = unary_tokens.shape[0]
    
    descendant = False # True when current node is a descendant of operator

    # For each action:
    for l in range(L):
        
        action = actions[0,l]
        
        if a_in_b(action, effector_tokens, E):
            # Does action match a target token?
            if descendant:
                # a token was found previously, but
                # we are still in a dangling node, therefore
                # we have a token inside a token expression
                # that we are not allowed to have e.g.
                # sin(sin(x)) .
                
                return True
            descendant  = True
            dangling    = 1
        elif descendant:
            if a_in_b(action, binary_tokens, B):
                # Does action match a binary token?
                # Then add one to dangling.
                dangling += 1
            elif a_not_in_b(action, unary_tokens, U):
                # Does action match a terminal token?
                # Then subtract one from dangling.
                # We skip the instance of unary since 
                # this leaves dangling to be += 0 
                # and assume that any token not binary 
                # and unary is a terminal. 
                dangling -= 1
            
            # If we no longer have any dangling nodes, 
            # Then we cannot be a descendant.     
            if dangling == 0:
                descendant = False
                
    return False  

@jit(nopython=True, parallel=False)
def jit_check_constraint_violation_descendant_with_target_tokens(\
        actions, target_tokens, effector_tokens, binary_tokens, unary_tokens):

    r"""
    
    Parameters
    __________

    actions : np.ndarray, shape=(1, L), dtype=np.int32
        Batch of action sequences. Values correspond to library indices.
        
    target_tokens : np.ndarray, dtype=np.int32
        Array of constraint tokens to match action against.

    binrary_tokens : np.ndarray, dtype=np.int32
        Array of binary function tokens in the current library. 
        
    uniary_tokens : np.ndarray, dtype=np.int32
        Array of unary function tokens in the current library. 

    Returns
    _______

    bool : Was the constraint violated. 
    
    """

    # Is this token item A found in the list of tokens in B?
    def a_in_b(a, B_tokens, B):
        for b in range(B):
            if a == B_tokens[b]:
                return True
        return False
    
    # Is this token item A NOT found in the list of tokens in B?                        
    def a_not_in_b(a, B_tokens, B):
        for b in range(B):
            if a == B_tokens[b]:
                return False
        return True
            
    _,L     = actions.shape
    T       = target_tokens.shape[0]
    B       = binary_tokens.shape[0]
    U       = unary_tokens.shape[0]
    
    descendant = False # True when current node is a descendant of operator

    # For each action:
    for l in range(L):
        
        action = actions[0,l]
        
        if a_in_b(action, effector_tokens, T):
            # Does action match a target token?
            descendant  = True
            dangling    = 1
        elif a_in_b(action, target_tokens, T):
            if descendant:                
                return True
        elif descendant:
            if a_in_b(action, binary_tokens, B):
                # Does action match a binary token?
                # Then add one to dangling.
                dangling += 1
            elif a_not_in_b(action, unary_tokens, U):
                # Does action match a terminal token?
                # Then subtract one from dangling.
                # We skip the instance of unary since 
                # this leaves dangling to be += 0 
                # and assume that any token not binary 
                # and unary is a terminal. 
                dangling -= 1
            
            # If we no longer have any dangling nodes, 
            # Then we cannot be a descendant.     
            if dangling == 0:
                descendant = False
                
    return False  

    

