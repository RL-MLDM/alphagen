from abc import ABC, abstractmethod

import tensorflow as tf

from dso.program import Program


class StateManager(ABC):
    """
    An interface for handling the tf.Tensor inputs to the Policy.
    """

    def setup_manager(self, policy):
        """
        Function called inside the policy to perform the needed initializations (e.g., if the tf context is needed)
        :param policy the policy class
        """
        self.policy = policy
        self.max_length = policy.max_length

    @abstractmethod
    def get_tensor_input(self, obs):
        """
        Convert an observation from a Task into a Tesnor input for the
        Policy, e.g. by performing one-hot encoding or embedding lookup.

        Parameters
        ----------
        obs : np.ndarray (dtype=np.float32)
            Observation coming from the Task.

        Returns
        --------
        input_ : tf.Tensor (dtype=tf.float32)
            Tensor to be used as input to the Policy.
        """
        return

    def process_state(self, obs):
        """
        Entry point for adding information to the state tuple.
        If not overwritten, this functions does nothing
        """
        return obs


def make_state_manager(config):
    """
    Parameters
    ----------
    config : dict
        Parameters for this StateManager.

    Returns
    -------
    state_manager : StateManager
        The StateManager to be used by the policy.
    """
    manager_dict = {
        "hierarchical": HierarchicalStateManager
    }

    if config is None:
        config = {}

    # Use HierarchicalStateManager by default
    manager_type = config.pop("type", "hierarchical")

    manager_class = manager_dict[manager_type]
    state_manager = manager_class(**config)

    return state_manager


class HierarchicalStateManager(StateManager):
    """
    Class that uses the previous action, parent, sibling, and/or dangling as
    observations.
    """

    def __init__(self, observe_parent=True, observe_sibling=True,
                 observe_action=False, observe_dangling=False, embedding=False,
                 embedding_size=8):
        """
        Parameters
        ----------
        observe_parent : bool
            Observe the parent of the Token being selected?

        observe_sibling : bool
            Observe the sibling of the Token being selected?

        observe_action : bool
            Observe the previously selected Token?

        observe_dangling : bool
            Observe the number of dangling nodes?

        embedding : bool
            Use embeddings for categorical inputs?

        embedding_size : int
            Size of embeddings for each categorical input if embedding=True.
        """
        self.observe_parent = observe_parent
        self.observe_sibling = observe_sibling
        self.observe_action = observe_action
        self.observe_dangling = observe_dangling
        self.library = Program.library

        # Parameter assertions/warnings
        assert self.observe_action + self.observe_parent + self.observe_sibling + self.observe_dangling > 0, \
            "Must include at least one observation."

        self.embedding = embedding
        self.embedding_size = embedding_size

    def setup_manager(self, policy):
        super().setup_manager(policy)
        # Create embeddings if needed
        if self.embedding:
            initializer = tf.random_uniform_initializer(minval=-1.0,
                                                        maxval=1.0,
                                                        seed=0)
            with tf.variable_scope("embeddings", initializer=initializer):
                if self.observe_action:
                    self.action_embeddings = tf.get_variable("action_embeddings",
                                                             [self.library.n_action_inputs, self.embedding_size],
                                                             trainable=True)
                if self.observe_parent:
                    self.parent_embeddings = tf.get_variable("parent_embeddings",
                                                             [self.library.n_parent_inputs, self.embedding_size],
                                                             trainable=True)
                if self.observe_sibling:
                    self.sibling_embeddings = tf.get_variable("sibling_embeddings",
                                                              [self.library.n_sibling_inputs, self.embedding_size],
                                                              trainable=True)

    def get_tensor_input(self, obs):
        observations = []
        unstacked_obs = tf.unstack(obs, axis=1)
        action, parent, sibling, dangling = unstacked_obs[:4]

        # Cast action, parent, sibling to int for embedding_lookup or one_hot
        action = tf.cast(action, tf.int32)
        parent = tf.cast(parent, tf.int32)
        sibling = tf.cast(sibling, tf.int32)

        # Action, parent, and sibling inputs are either one-hot or embeddings
        if self.observe_action:
            if self.embedding:
                x = tf.nn.embedding_lookup(self.action_embeddings, action)
            else:
                x = tf.one_hot(action, depth=self.library.n_action_inputs)
            observations.append(x)
        if self.observe_parent:
            if self.embedding:
                x = tf.nn.embedding_lookup(self.parent_embeddings, parent)
            else:
                x = tf.one_hot(parent, depth=self.library.n_parent_inputs)
            observations.append(x)
        if self.observe_sibling:
            if self.embedding:
                x = tf.nn.embedding_lookup(self.sibling_embeddings, sibling)
            else:
                x = tf.one_hot(sibling, depth=self.library.n_sibling_inputs)
            observations.append(x)

        # Dangling input is just the value of dangling
        if self.observe_dangling:
            x = tf.expand_dims(dangling, axis=-1)
            observations.append(x)

        input_ = tf.concat(observations, -1)
        # possibly concatenates additional observations (e.g., bert embeddings)
        if len(unstacked_obs) > 4:
            input_ = tf.concat([input_, tf.stack(unstacked_obs[4:], axis=-1)], axis=-1)
        return input_
