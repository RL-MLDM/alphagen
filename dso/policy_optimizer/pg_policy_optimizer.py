
import tensorflow as tf
from dso.policy_optimizer import PolicyOptimizer
from dso.policy import Policy

class PGPolicyOptimizer(PolicyOptimizer):
    """Vanilla policy gradient policy optimizer.

    Parameters
    ----------
    cell : str
        Recurrent cell to use. Supports 'lstm' and 'gru'.

    num_layers : int
        Number of RNN layers.

    num_units : int or list of ints
        Number of RNN cell units in each of the RNN's layers. If int, the value
        is repeated for each layer. 

    initiailizer : str
        Initializer for the recurrent cell. Supports 'zeros' and 'var_scale'.
        
    """
    def __init__(self, 
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
        super()._setup_policy_optimizer(sess, policy, debug, summary, optimizer, learning_rate, entropy_weight, entropy_gamma)


    def _set_loss(self):
        with tf.name_scope("losses"):
            # Retrieve rewards from batch
            r = self.sampled_batch_ph.rewards
            # Baseline is the worst of the current samples r
            self.pg_loss = tf.reduce_mean((r - self.baseline) * self.neglogp, name="pg_loss")
            # Loss already is set to entropy loss
            self.loss += self.pg_loss


    def _preppend_to_summary(self):
        with tf.name_scope("summary"):
            tf.summary.scalar("pg_loss", self.pg_loss)


    def train_step(self, baseline, sampled_batch):
        """Computes loss, trains model, and returns summaries."""
        feed_dict = {
            self.baseline : baseline,
            self.sampled_batch_ph : sampled_batch
        }

        summaries, _ = self.sess.run([self.summaries, self.train_op], feed_dict=feed_dict)

        return summaries