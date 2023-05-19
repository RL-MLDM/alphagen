from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from dso.program import Program
from dso.memory import Batch
from dso.utils import import_custom_source
from dso.policy.policy import Policy
from dso.utils import make_batch_ph

# Used for function annotations using the type system
summaries = tf.TensorArray

def make_policy_optimizer(sess, policy, policy_optimizer_type, **config_policy_optimizer):
    """Factory function for policy optimizer object."""

    if policy_optimizer_type == "pg":
        from dso.policy_optimizer.pg_policy_optimizer import PGPolicyOptimizer
        policy_optimizer_class = PGPolicyOptimizer
    elif policy_optimizer_type == "pqt":
        from dso.policy_optimizer.pqt_policy_optimizer import PQTPolicyOptimizer
        policy_optimizer_class = PQTPolicyOptimizer
    elif policy_optimizer_type == "ppo":
        from dso.policy_optimizer.ppo_policy_optimizer import PPOPolicyOptimizer
        policy_optimizer_class = PPOPolicyOptimizer
    else:
        # Custom policy import
        policy_optimizer_class = import_custom_source(policy_optimizer_type)
        assert issubclass(policy_optimizer_class, Policy), \
                "Custom policy {} must subclass dso.policy.Policy.".format(policy_optimizer_class)
        
    policy_optimizer = policy_optimizer_class(sess,
                                              policy,
                                              **config_policy_optimizer)

    return policy_optimizer

class PolicyOptimizer(ABC):
    """Abstract class for a policy optimizer. A policy optimizer is an 
    algorithm for optimizing the parameters of a parametrized policy.

    To define a new optimizer, inherit from this class and add the following
    methods (look in _setup_policy_optimizer below):

        _set_loss() : Define the \propto \log(p(\tau|\theta)) loss for the method
        _preppend_to_summary() : Add additional fields for the tensorflow summary

    """    

    def _init(self, 
            sess : tf.Session,
            policy : Policy,
            debug : int = 0,    
            summary : bool = False,
            # Optimizer hyperparameters
            optimizer : str = 'adam',
            learning_rate : float = 0.001,
            # Loss hyperparameters
            entropy_weight : float = 0.005,
            entropy_gamma : float = 1.0) -> None:
        '''Parameters
        ----------
        sess : tf.Session
            TensorFlow Session object.

        policy : dso.policy.Policy
            Parametrized probability distribution over discrete objects

        debug : int
            Debug level, also used in learn(). 0: No debug. 1: Print shapes and
            number of parameters for each variable.

        summary : bool
            Write tensorboard summaries?

        optimizer : str
            Optimizer to use. Supports 'adam', 'rmsprop', and 'sgd'.

        learning_rate : float
            Learning rate for optimizer.

        entropy_weight : float
            Coefficient for entropy bonus.

        entropy_gamma : float or None
            Gamma in entropy decay. None (or
            equivalently, 1.0) turns off entropy decay.
        '''    
        self.sess = sess
        self.policy = policy

        # Needed in _setup_optimizer
        self.debug = debug
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        # Need in self.summary
        self.summary = summary
        
        # Needed for make_batch_ph calls 
        self.n_choices = Program.library.L

        # Placeholders, computed after instantiating expressions
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=(), name="batch_size")
        self.baseline = tf.placeholder(dtype=tf.float32, shape=(), name="baseline")
   
        # On policy batch
        self.sampled_batch_ph = make_batch_ph("sampled_batch", self.n_choices)

        # Need in _init_loss_with_entropy
        self.entropy_weight = entropy_weight
        self.entropy_gamma = entropy_gamma


    def _init_loss_with_entropy(self) -> None:
        # Add entropy contribution to loss. The entropy regularizer does not
        # depend on the particular policy optimizer
        with tf.name_scope("losses"):

            self.neglogp, entropy = self.policy.make_neglogp_and_entropy(self.sampled_batch_ph, self.entropy_gamma)

            # Entropy loss
            self.entropy_loss = -self.entropy_weight * tf.reduce_mean(entropy, name="entropy_loss")
            loss = self.entropy_loss

            # self.loss is modified in the child object
            self.loss = loss


    @abstractmethod
    def _set_loss(self) -> None:
        """Define the \propto \log(p(\tau|\theta)) loss for the method

        Returns
        -------
            None
        """
        raise NotImplementedError


    def _setup_optimizer(self):
        """ Setup the optimizer
        """    
        def make_optimizer(name, learning_rate):
            if name == "adam":
                return tf.train.AdamOptimizer(learning_rate=learning_rate)
            if name == "rmsprop":
                return tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99)
            if name == "sgd":
                return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            raise ValueError("Did not recognize optimizer '{}'".format(name))

        # Create training op
        optimizer = make_optimizer(name=self.optimizer, learning_rate=self.learning_rate)
        with tf.name_scope("train"):
            self.grads_and_vars = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(self.grads_and_vars)
            # The two lines above are equivalent to:
            # self.train_op = optimizer.minimize(self.loss)
        with tf.name_scope("grad_norm"):
            self.grads, _ = list(zip(*self.grads_and_vars))
            self.norms = tf.global_norm(self.grads)  
        
        if self.debug >= 1:
            total_parameters = 0
            print("")
            for variable in tf.trainable_variables():
                shape = variable.get_shape()
                n_parameters = np.product(shape)
                total_parameters += n_parameters
                print("Variable:    ", variable.name)
                print("  Shape:     ", shape)
                print("  Parameters:", n_parameters)
            print("Total parameters:", total_parameters)


    # abstractmethod (override if needed)
    def _preppend_to_summary(self) -> None:
        """Add particular fields to the summary log.
        Override if needed.
        """
        pass
        

    def _setup_summary(self) -> None:
        """ Setup tensor flow summary
        """    
        with tf.name_scope("summary"):
            tf.summary.scalar("entropy_loss", self.entropy_loss)
            tf.summary.scalar("total_loss", self.loss)
            tf.summary.scalar("reward", tf.reduce_mean(self.sampled_batch_ph.rewards))
            tf.summary.scalar("baseline", self.baseline)
            tf.summary.histogram("reward", self.sampled_batch_ph.rewards)
            tf.summary.histogram("length", self.sampled_batch_ph.lengths)
            for g, v in self.grads_and_vars:
                tf.summary.histogram(v.name, v)
                tf.summary.scalar(v.name + '_norm', tf.norm(v))
                tf.summary.histogram(v.name + '_grad', g)
                tf.summary.scalar(v.name + '_grad_norm', tf.norm(g))
            tf.summary.scalar('gradient norm', self.norms)
            self.summaries = tf.summary.merge_all()


    def _setup_policy_optimizer(self, 
            sess : tf.Session,
            policy : Policy,
            debug : int = 0,    
            summary : bool = False,
            # Optimizer hyperparameters
            optimizer : str = 'adam',
            learning_rate : float = 0.001,
            # Loss hyperparameters
            entropy_weight : float = 0.005,
            entropy_gamma : float = 1.0) -> None:
        """Setup of the policy optimizer.
        """ 
        self._init(sess, policy, debug, summary, optimizer, learning_rate, entropy_weight, entropy_gamma)
        self._init_loss_with_entropy()
        self._set_loss() # Abstract method defined in derived class
        self._setup_optimizer()
        if self.summary:
            self._preppend_to_summary() # Abstract method defined in derived class
            self._setup_summary()
        else:
            self.summaries = tf.no_op()        


    @abstractmethod
    def train_step(self, 
            baseline : np.ndarray, 
            sampled_batch : Batch) -> summaries:
        """Computes loss, trains model, and returns summaries.

        Returns
        -------
            None
        """
        raise NotImplementedError



