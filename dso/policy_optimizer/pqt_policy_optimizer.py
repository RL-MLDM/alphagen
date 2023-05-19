
import tensorflow as tf
from dso.policy_optimizer import PolicyOptimizer
from dso.policy import Policy
from dso.utils import make_batch_ph

class PQTPolicyOptimizer(PolicyOptimizer):
    """Priority Queue Ttraining policy gradient policy optimizer.

    Parameters
    ----------
     pqt_k : int
        Size of priority queue.

    pqt_batch_size : int
        Size of batch to sample (with replacement) from priority queue.

    pqt_weight : float
        Coefficient for PQT loss function.

    pqt_use_pg : bool
        Use policy gradient loss when using PQT?   
        
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
            entropy_gamma : float = 1.0,
            # PQT hyperparameters
            pqt_k : int = 10,
            pqt_batch_size : int = 1,
            pqt_weight : float = 200.0,
            pqt_use_pg: bool = False) -> None:
        self.pqt_k = pqt_k
        self.pqt_batch_size = pqt_batch_size
        self.pqt_weight = pqt_weight
        self. pqt_use_pg = pqt_use_pg
        super()._setup_policy_optimizer(sess, policy, debug, summary, optimizer, learning_rate, entropy_weight, entropy_gamma)


    def _set_loss(self):
        with tf.name_scope("losses"):
            # Create placeholder for PQT batch
            self.pqt_batch_ph = make_batch_ph("pqt_batch", self.n_choices) #self.n_choices is defined in parent class

            pqt_neglogp, _ = self.policy.make_neglogp_and_entropy(self.pqt_batch_ph, self.entropy_gamma)
            self.pqt_loss = self.pqt_weight * tf.reduce_mean(pqt_neglogp, name="pqt_loss")
        
            # Loss already is set to entropy loss
            self.loss += self.pqt_loss


    def _preppend_to_summary(self):
        with tf.name_scope("summary"):
            tf.summary.scalar("pqt_loss", self.pqt_loss)


    def train_step(self, baseline, sampled_batch, pqt_batch):
        feed_dict = {
            self.baseline : baseline,
            self.sampled_batch_ph : sampled_batch
        }
        feed_dict.update({
                self.pqt_batch_ph : pqt_batch
        })

        summaries, _ = self.sess.run([self.summaries, self.train_op], feed_dict=feed_dict)

        return summaries