"""Controller used to generate distribution over hierarchical, variable-length objects."""
import tensorflow as tf
import numpy as np

from dso.program import Program
from dso.program import _finish_tokens
from dso.memory import Batch

from dso.policy import Policy
from dso.utils import make_batch_ph

class LinearWrapper(tf.contrib.rnn.LayerRNNCell):
    """RNNCell wrapper that adds a linear layer to the output.

    See: https://github.com/tensorflow/models/blob/master/research/brain_coder/single_task/pg_agent.py
    """

    def __init__(self, cell, output_size):
        self.cell = cell
        self._output_size = output_size

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(type(self).__name__):
            outputs, state = self.cell(inputs, state, scope=scope)
            logits = tf.layers.dense(outputs, units=self._output_size)

        return logits, state

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self.cell.state_size

    def zero_state(self, batch_size, dtype):
        return self.cell.zero_state(batch_size, dtype)

def safe_cross_entropy(p, logq, axis=-1):
    """Compute p * logq safely, by susbstituting
    logq[index] = 1 for index such that p[index] == 0
    """
    # Put 1 where p == 0. In the case, q =p, logq = -inf and this
    # might procude numerical errors below
    safe_logq = tf.where(tf.equal(p, 0.), tf.ones_like(logq), logq)
    # Safely compute the product
    return - tf.reduce_sum(p * safe_logq, axis)

class RNNPolicy(Policy):
    """Recurrent neural network (RNN) policy used to generate expressions.

    Specifically, the RNN outputs a distribution over pre-order traversals of
    symbolic expression trees.

    Parameters
    ----------
    action_prob_lowerbound: float
        Lower bound on probability of each action.

    cell : str
        Recurrent cell to use. Supports 'lstm' and 'gru'.

    max_attempts_at_novel_batch: int
        maximum number of repetitions of sampling to get b new samples
        during a call of policy.sample(b)

    num_layers : int
        Number of RNN layers.

    num_units : int or list of ints
        Number of RNN cell units in each of the RNN's layers. If int, the value
        is repeated for each layer. 

    sample_novel_batch: bool
        if True, then a call to policy.sample(b) attempts to produce b samples
        that are not contained in the cache

    initiailizer : str
        Initializer for the recurrent cell. Supports 'zeros' and 'var_scale'.
        
    """
    def __init__(self, sess, prior, state_manager, 
                 debug = 0,
                 max_length = 30,
                 action_prob_lowerbound = 0.0,
                 max_attempts_at_novel_batch = 10,
                 sample_novel_batch = False,
                 # RNN cell hyperparameters
                 cell ='lstm',
                 num_layers=1,
                 num_units=32,
                 initializer='zeros'):
        super().__init__(sess, prior, state_manager, debug, max_length)
        
        assert 0 <= action_prob_lowerbound  and action_prob_lowerbound <= 1
        self.action_prob_lowerbound = action_prob_lowerbound

        # len(tokens) in library
        self.n_choices = Program.library.L

        # Placeholders, computed after instantiating expressions
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=(), name="batch_size")

        # setup model
        self._setup_tf_model(cell, num_layers, num_units, initializer)

        self.max_attempts_at_novel_batch = max_attempts_at_novel_batch
        self.sample_novel_batch = sample_novel_batch

    def _setup_tf_model(
            self, 
            cell ='lstm',
            num_layers=1,
            num_units=32,
            initializer='zeros'):

        # Defined in super class
        # This can be susbtituted below
        n_choices = self.n_choices
        prior = self.prior
        state_manager = self.state_manager
        max_length = self.max_length

        # Build RNN policy
        with tf.name_scope("controller"):

            def make_initializer(name):
                if name == "zeros":
                    return tf.zeros_initializer()
                if name == "var_scale":
                    return tf.contrib.layers.variance_scaling_initializer(
                            factor=0.5, mode='FAN_AVG', uniform=True, seed=0)
                raise ValueError("Did not recognize initializer '{}'".format(name))

            def make_cell(name, num_units, initializer):
                if name == 'lstm':
                    return tf.nn.rnn_cell.LSTMCell(num_units, initializer=initializer)
                if name == 'gru':
                    return tf.nn.rnn_cell.GRUCell(num_units, kernel_initializer=initializer, bias_initializer=initializer)
                raise ValueError("Did not recognize cell type '{}'".format(name))

            # Create recurrent cell
            if isinstance(num_units, int):
                num_units = [num_units] * num_layers
            initializer = make_initializer(initializer)
            cell = tf.contrib.rnn.MultiRNNCell(
                    [make_cell(cell, n, initializer=initializer) for n in num_units])
            cell = LinearWrapper(cell=cell, output_size=n_choices)

            # Set the cell attribute needed for make_neglog_probs_and_entropy
            self.cell = cell

            task = Program.task
            initial_obs = task.reset_task(prior)
            state_manager.setup_manager(self)
            initial_obs = tf.broadcast_to(initial_obs, [self.batch_size, len(initial_obs)]) # (?, obs_dim)
            initial_obs = state_manager.process_state(initial_obs)

            # Get initial prior
            initial_prior = self.prior.initial_prior()
            initial_prior = tf.constant(initial_prior, dtype=tf.float32)
            initial_prior = tf.broadcast_to(initial_prior, [self.batch_size, n_choices])

            def loop_fn(time, cell_output, cell_state, loop_state):

                if cell_output is None: # time == 0
                    finished = tf.zeros(shape=[self.batch_size], dtype=tf.bool)
                    obs = initial_obs
                    next_input = state_manager.get_tensor_input(obs)
                    next_cell_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32) # 2-tuple, each shape (?, num_units)
                    emit_output = None
                    actions_ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False) # Read twice
                    obs_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
                    priors_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=True)
                    prior = initial_prior
                    #lengths = tf.ones(shape=[self.batch_size], dtype=tf.int32)
                    next_loop_state = (
                        actions_ta,
                        obs_ta,
                        priors_ta,
                        obs,
                        prior,
                        finished)
                else:
                    actions_ta, obs_ta, priors_ta, obs, prior, finished = loop_state
                    # apply bound to logits before applying prior, so that hard constraints
                    # are respected
                    if self.action_prob_lowerbound != 0.0:
                        cell_output = self.apply_action_prob_lowerbound(cell_output)
                    logits = cell_output + prior
                    next_cell_state = cell_state
                    emit_output = logits

                    # Sample action
                    action = tf.random.categorical(logits=logits, num_samples=1,
                                                   dtype=tf.int32, seed=1)[:, 0]
                    next_actions_ta = actions_ta.write(time - 1, action) # Write chosen actions
                    actions = tf.transpose(next_actions_ta.stack())  # Shape: (?, time)

                    # Compute obs and prior
                    next_obs, next_prior, next_finished = tf.py_func(func=task.get_next_obs,
                                                                     inp=[actions, obs, finished],
                                                                     Tout=[tf.float32, tf.float32, tf.bool])
                    next_prior.set_shape([None, n_choices])
                    next_obs.set_shape([None, task.OBS_DIM])
                    next_finished.set_shape([None])
                    next_obs = state_manager.process_state(next_obs)
                    next_input = state_manager.get_tensor_input(next_obs)
                    next_obs_ta = obs_ta.write(time - 1, obs) # Write OLD obs
                    next_priors_ta = priors_ta.write(time - 1, prior) # Write OLD prior
                    finished = next_finished = tf.logical_or(
                        next_finished,
                        time >= max_length)
                    next_loop_state = (next_actions_ta,
                                       next_obs_ta,
                                       next_priors_ta,
                                       next_obs,
                                       next_prior,
                                       next_finished)

                return (finished, next_input, next_cell_state, emit_output, next_loop_state)

            # Returns RNN emit outputs TensorArray (i.e. logits), final cell state, and final loop state
            with tf.variable_scope('policy'):
                _, _, loop_state = tf.nn.raw_rnn(cell=cell, loop_fn=loop_fn)
                actions_ta, obs_ta, priors_ta, _, _, _ = loop_state

            self.actions = tf.transpose(actions_ta.stack(), perm=[1, 0]) # (?, max_length)
            self.obs = tf.transpose(obs_ta.stack(), perm=[1, 2, 0]) # (?, obs_dim, max_length)
            self.priors = tf.transpose(priors_ta.stack(), perm=[1, 0, 2]) # (?, max_length, n_choices)
            
            # Memory batch
            self.memory_batch_ph = make_batch_ph("memory_batch", n_choices)
            memory_neglogp, _ = self.make_neglogp_and_entropy(self.memory_batch_ph, None)

            self.memory_probs = tf.exp(-memory_neglogp)
            self.memory_logps = -memory_neglogp
            

    def make_neglogp_and_entropy(self, B, entropy_gamma) :
        """Computes the negative log-probabilities for a given
        batch of actions, observations and priors
        under the current policy.

        Returns
        -------
        neglogp, entropy :
            Tensorflow tensors
        """

        # Entropy decay vector
        if entropy_gamma is None:
            entropy_gamma = 1.0
        entropy_gamma_decay = np.array([entropy_gamma**t for t in range(self.max_length)], dtype=np.float32)

        with tf.variable_scope('policy', reuse=True):
            logits, _ = tf.nn.dynamic_rnn(cell=self.cell,
                                          inputs=self.state_manager.get_tensor_input(B.obs),
                                          sequence_length=B.lengths, # Backpropagates only through sequence length
                                          dtype=tf.float32)

        if self.action_prob_lowerbound != 0.0:
            logits = self.apply_action_prob_lowerbound(logits)

        logits += B.priors
        probs = tf.nn.softmax(logits)
        logprobs = tf.nn.log_softmax(logits)
        B_max_length = tf.shape(B.actions)[1] # Maximum sequence length for this Batch
        # Generate mask from sequence lengths
        # NOTE: Using this mask for neglogp and entropy actually does NOT
        # affect training because gradients are zero outside the lengths.
        # However, the mask makes tensorflow summaries accurate.
        mask = tf.sequence_mask(B.lengths, maxlen=B_max_length, dtype=tf.float32)
        # Negative log probabilities of sequences
        actions_one_hot = tf.one_hot(B.actions, depth=self.n_choices, axis=-1, dtype=tf.float32)
        neglogp_per_step = safe_cross_entropy(actions_one_hot, logprobs, axis=2) # Sum over action dim
        neglogp = tf.reduce_sum(neglogp_per_step * mask, axis=1) # Sum over time dim
        
        # NOTE 1: The above implementation is the same as the one below:
        # neglogp_per_step = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=actions)
        # neglogp = tf.reduce_sum(neglogp_per_step, axis=1) # Sum over time
        # NOTE 2: The above implementation is also the same as the one below, with a few caveats:
        #   Exactly equivalent when removing priors.
        #   Equivalent up to precision when including clipped prior.
        #   Crashes when prior is not clipped due to multiplying zero by -inf.
        # neglogp_per_step = -tf.nn.log_softmax(logits + tf.clip_by_value(priors, -2.4e38, 0)) * actions_one_hot
        # neglogp_per_step = tf.reduce_sum(neglogp_per_step, axis=2)
        # neglogp = tf.reduce_sum(neglogp_per_step, axis=1) # Sum over time
        
        # If entropy_gamma = 1, entropy_gamma_decay_mask == mask
        sliced_entropy_gamma_decay = tf.slice(entropy_gamma_decay, [0], [B_max_length])
        entropy_gamma_decay_mask = sliced_entropy_gamma_decay * mask # ->(batch_size, max_length)
        entropy_per_step = safe_cross_entropy(probs, logprobs, axis=2) # Sum over action dim -> (batch_size, max_length)
        entropy = tf.reduce_sum(entropy_per_step * entropy_gamma_decay_mask, axis=1) # Sum over time dim -> (batch_size, )
        
        return neglogp, entropy


    def sample(self, n : int) :
        """Sample batch of n expressions

        Returns
        -------
        actions, obs, priors : 
            Or a batch
        """
        if self.sample_novel_batch:
            actions, obs, priors = self.sample_novel(n)
        else:
            feed_dict = {self.batch_size : n}
            actions, obs, priors = self.sess.run(
                [self.actions, self.obs, self.priors], feed_dict=feed_dict)

        return actions, obs, priors

    def sample_novel(self, n: int):
        """Sample a batch of n expressions not contained in cache.

        If unable to do so within self.max_attempts_at_novel_batch,
        then fills in the remaining slots with previously-seen samples.

        Parameters
        ----------
        n: int
            batch size

        Returns
        -------
        unique_a, unique_o, unique_p: np.ndarrays
        """
        feed_dict = {self.batch_size : n}
        n_novel = 0
        # Keep the samples that are produced by policy and already exist in cache,
        # so that DSO can train on everything
        old_a, old_o, old_p = [], [], []
        # Store the new samples separately for (expensive) reward evaluation
        new_a, new_o, new_p = [], [], []
        n_attempts = 0
        while n_novel < n and n_attempts < self.max_attempts_at_novel_batch:
            # [batch, time], [batch, obs_dim, time], [batch, time, n_choices]
            actions, obs, priors = self.sess.run(
                [self.actions, self.obs, self.priors], feed_dict=feed_dict)
            n_attempts += 1
            new_indices = [] # indices of new and unique samples
            old_indices = [] # indices of samples already in cache
            for idx, a in enumerate(actions):
                # tokens = Program._finish_tokens(a)
                tokens = _finish_tokens(a)
                key = tokens.tostring()
                if not key in Program.cache.keys() and n_novel < n:
                    new_indices.append(idx)
                    n_novel += 1
                if key in Program.cache.keys():
                    old_indices.append(idx)
            # get all new actions, obs, priors in this group
            new_a.append(np.take(actions, new_indices, axis=0))
            new_o.append(np.take(obs, new_indices, axis=0))
            new_p.append(np.take(priors, new_indices, axis=0))
            old_a.append(np.take(actions, old_indices, axis=0))
            old_o.append(np.take(obs, old_indices, axis=0))
            old_p.append(np.take(priors, old_indices, axis=0))

        # number of slots in batch to be filled in by redundant samples
        n_remaining = n - n_novel

        # -------------------- combine all -------------------- #
        # Pad everything to max_length
        for tup, name in zip([(old_a, new_a), (old_o, new_o), (old_p, new_p)],
                                    ['action', 'obs', 'prior']):
            dim_length = 1 if name in ['action', 'prior'] else 2
            max_length = np.max([list_batch.shape[dim_length] for
                                 list_batch in tup[0] + tup[1]])
            # tup is a tuple of (old_?, new_?), each is a list of batches
            for list_batch in tup:
                for idx, batch in enumerate(list_batch):
                    n_pad = max_length - batch.shape[dim_length]
                    # Pad with 0 for everything because training step
                    # truncates based on each sample's own sequence length
                    # so the value does not matter
                    if name == 'action':
                        width = ((0,0),(0,n_pad))
                        vals = ((0,0),(0,0))
                    elif name == 'obs':
                        width = ((0,0),(0,0),(0,n_pad))
                        vals = ((0,0),(0,0),(0,0))
                    else:
                        width = ((0,0),(0,n_pad),(0,0))
                        vals = ((0,0),(0,0),(0,0))
                    list_batch[idx] = np.pad(
                        batch, pad_width=width, mode='constant',
                        constant_values=vals)

        old_a = np.concatenate(old_a)
        old_o = np.concatenate(old_o)
        old_p = np.concatenate(old_p)
        # If not enough novel samples, then fill in with redundancies
        new_a = np.concatenate(new_a + [old_a[:n_remaining]])
        new_o = np.concatenate(new_o + [old_o[:n_remaining]])
        new_p = np.concatenate(new_p + [old_p[:n_remaining]])

        # first entry serves to force object type, and also
        # indicates not to use it if zero
        self.extended_batch = np.array(
            [old_a.shape[0], old_a, old_o, old_p], dtype=object)
        self.valid_extended_batch = True

        return new_a, new_o, new_p

    def compute_probs(self, memory_batch, log=False):
        """Compute the probabilities of a Batch."""

        feed_dict = {
            self.memory_batch_ph : memory_batch
        }

        if log:
            fetch = self.memory_logps
        else:
            fetch = self.memory_probs
        probs = self.sess.run([fetch], feed_dict=feed_dict)[0]
        return probs

    def apply_action_prob_lowerbound(self, logits):
        """Applies a lower bound to probabilities of each action.

        Parameters
        ----------
        logits: tf.Tensor where last dimension has size self.n_choices

        Returns
        -------
        logits_bounded: tf.Tensor
        """
        probs = tf.nn.softmax(logits, axis=-1)
        probs_bounded = ((1-self.action_prob_lowerbound)*probs +
                         self.action_prob_lowerbound/
                         float(self.n_choices))
        logits_bounded = tf.log(probs_bounded)

        return logits_bounded
