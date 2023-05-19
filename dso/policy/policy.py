from abc import ABC, abstractmethod

from typing import Tuple, TypeVar

import tensorflow as tf
import dso
from dso.prior import LengthConstraint
from dso.program import Program
from dso.utils import import_custom_source
from dso.prior import JointPrior
from dso.tf_state_manager import StateManager
from dso.memory import Batch

# Used for function annotations using the type system
actions = tf.TensorArray
obs     = tf.TensorArray
priors  = tf.TensorArray
neglogp = tf.TensorArray
entropy = tf.TensorArray

def make_policy(sess, prior, state_manager, policy_type, **config_policy):
    """Factory function for Policy object."""

    if policy_type == "rnn":
        from dso.policy.rnn_policy import RNNPolicy
        policy_class = RNNPolicy
    else:
        # Custom policy import
        policy_class = import_custom_source(policy_type)
        assert issubclass(policy_class, Policy), \
                "Custom policy {} must subclass dso.policy.Policy.".format(policy_class)
        
    policy = policy_class(sess,
                          prior,
                          state_manager,
                          **config_policy)

    return policy

class Policy(ABC):
    """Abstract class for a policy. A policy is a parametrized probability 
    distribution over discrete objects. DSO algorithms optimize the parameters 
    of this distribution to generate discrete objects with high rewards.
    """    

    def __init__(self, 
            sess : tf.Session,
            prior : JointPrior,
            state_manager : StateManager,
            debug : int = 0,  
            max_length : int = 30) -> None:
        '''Parameters
        ----------
        sess : tf.Session
            TenorFlow Session object.
    
        prior : dso.prior.JointPrior
            JointPrior object used to adjust probabilities during sampling.
    
        state_manager: dso.tf_state_manager.StateManager
            Object that handles the state features to be used
        
        debug : int
            Debug level, also used in learn(). 0: No debug. 1: Print shapes and
            number of parameters for each variable.

        max_length : int or None
            Maximum sequence length. This will be overridden if a LengthConstraint
            with a maximum length is part of the prior.
        '''    
        self.sess = sess
        self.prior = prior
        self.state_manager = state_manager
        self.debug = debug

        # Set self.max_length depending on the Prior 
        self._set_max_length(max_length)

        # Samples produced during attempt to get novel samples.
        # Will be combined with checkpoint-loaded samples for next training step
        self.extended_batch = None
        self.valid_extended_batch = False
        
    def _set_max_length(self, max_length : int) -> None:
        """Set the max legnth depending on the Prior
        """
        # Find max_length from the LengthConstraint prior, if it exists
        # For binding task, max_length is # of allowed mutations or master-seq length
        # Both priors will never happen in the same experiment
        prior_max_length = None
        for single_prior in self.prior.priors:
            if isinstance(single_prior, LengthConstraint):
                if single_prior.max is not None:
                    prior_max_length = single_prior.max
                    self.max_length = prior_max_length
                break

        if prior_max_length is None:
            assert max_length is not None, "max_length must be specified if "\
                "there is no LengthConstraint."
            self.max_length = max_length
            print("WARNING: Maximum length not constrained. Sequences will "
                  "stop at {} and complete by repeating the first input "
                  "variable.".format(self.max_length))
        elif max_length is not None and max_length != self.max_length:
            print("WARNING: max_length ({}) will be overridden by value from "
                  "LengthConstraint ({}).".format(max_length, self.max_length))

    @abstractmethod
    def _setup_tf_model(self, **kwargs) -> None:
        """"Setup the TensorFlow graph(s).

        Returns
        -------
            None
        """
        raise NotImplementedError

    @abstractmethod
    def make_neglogp_and_entropy(self, 
            B : Batch,
            entropy_gamma : float
            ) -> Tuple[neglogp, entropy]:
        """Computes the negative log-probabilities for a given
        batch of actions, observations and priors
        under the current policy.
        
        Returns
        -------
        neglogp, entropy : 
            Tensorflow tensors        
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, n : int) -> Tuple[actions, obs, priors]:
        """Sample batch of n expressions.

        Returns
        -------
        actions, obs, priors : 
            Or a batch
        """
        raise NotImplementedError

    @abstractmethod
    def compute_probs(self, memory_batch, log=False):
        """Compute the probabilities of a Batch.

        Returns
        -------
        probs : 
            Or a batch
        """
        raise NotImplementedError


