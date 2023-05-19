import os
from datetime import datetime, timedelta

import tensorflow as tf
import numpy as np
import pandas as pd

from dso.program import Program, from_tokens


class Checkpoint():
    """
    A helper class to checkpoint models.

    Methods
    -------
    save
        Save a checkpoint.

    load
        Load from a given checkpoint.

    update
        Maybe save a checkpoint depending on frequency configuration.
    """

    def __init__(self, model, load_path=None, save_freq=23, units="hours",
                 save_on_done=False):
        """
        model : dso.DeepSymbolicOptimizer
            The model to checkpoint.

        load_path : str or None
            Path to initial checkpoint directory to load. If None, do not start from
            checkpoint.

        save_freq : float or None
            The frequency at which to save a checkpoint. If None, non-final checkpoints
            will not be automatically saved.

        units : str
            The units of save_freq. Supports "hours", "minutes", "seconds", "iterations".

        save_on_done : bool
            Whether to save a final checkpoint upon reaching model.trainer.done.
        """

        self.model = model
        if model.save_path is not None:
            self.checkpoint_dir = os.path.join(model.save_path, "checkpoint")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # Create the Saver
        self.saver = tf.train.Saver()

        # Load from existing checkpoint, if given
        if load_path is not None:
            self.load(load_path)

        # Setup time-based checkpointing
        if save_freq is not None and units in ["hours", "minutes", "seconds"]:
            if units == "hours":
                self.dt = timedelta(hours=save_freq)
            elif units == "minutes":
                self.dt = timedelta(minutes=save_freq)
            elif units == "seconds":
                self.dt = timedelta(seconds=save_freq)
            self.next_save_time = datetime.now() + self.dt
        else:
            self.next_save_time = None
            self.dt = None

        # Setup iteration-based checkpointing
        if save_freq is not None and units == "iterations":
            self.save_freq_iters = save_freq
        else:
            self.save_freq_iters = None

        self.save_on_done = save_on_done

    def update(self):
        """
        Maybe a save a checkpoint, depending on configuration. This should be called
        each iteration, i.e. after model.trainer.run_one_step().
        """

        # Save final checkpoint if done
        if self.save_on_done and self.model.trainer.done:
            self.save()

        # Save if time-based frequency is met
        elif self.next_save_time is not None and datetime.now() > self.next_save_time:
            self.save()
            self.next_save_time = datetime.now() + self.dt

        # Save if iteration-based frequency is met
        elif self.save_freq_iters is not None and (self.model.trainer.iteration % self.save_freq_iters) == 0:
            self.save()

    def save(self, save_path=None):
        """
        Save a checkpoint.

        Parameters
        ----------
        save_path : str or None
            Directory in which to save checkpoint. If None, save in:
            <self.model.save_path>/checkpoint/checkpoint_<timestamp>.
        """

        # Determine the save path
        if save_path is None:
            assert self.checkpoint_dir is not None, "Cannot support automated checkpointing with model.save_dir=None."
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            save_path = os.path.join(self.checkpoint_dir,
                                    "checkpoint_{}".format(timestamp))
        if os.path.exists(save_path):
            paths = os.listdir(os.path.dirname(save_path))
            paths = [path for path in paths if path.startswith(os.path.basename(save_path))]
            save_path += "_{}".format(len(paths))
        os.makedirs(save_path, exist_ok=False)

        # Save the TensorFlow graph
        # print("Saving TensorFlow graph...")
        tf_save_path = os.path.join(save_path, "tf")
        self.saver.save(self.model.sess, tf_save_path)

        # Save the Trainer
        # print("Saving Trainer...")
        trainer_save_path = os.path.join(save_path, "trainer.json")
        self.model.trainer.save(trainer_save_path)

        # Save the priority queue, if applicable
        # TBD: This should be in self.model.trainer.save or self.model.trainer.policy_optimizer.save after refactoring PolicyOptimizers to handle their own bookkeeping
        if self.model.trainer.priority_queue is not None:
            priority_queue_save_path = os.path.join(save_path, "priority_queue.npz")
            self.model.trainer.priority_queue.save(priority_queue_save_path)

        # Save the cache
        # print("Saving cache...")
        # TBD: Abstract into cache saving function
        cache_save_path = os.path.join(save_path, "cache.csv")        
        cache_programs = Program.cache.values()
        cache_tokens = [",".join(map(str, p.tokens.tolist())) for p in cache_programs]
        cache_rewards = [p.r for p in cache_programs]
        cache_data = { "tokens" : cache_tokens, "rewards" : cache_rewards }
        cache_df = pd.DataFrame(cache_data)
        cache_df.to_csv(cache_save_path, index=False)

        # Save the extra samples that were produced while attempting to
        # generate a batch of new and unique samples
        if self.model.trainer.policy.valid_extended_batch:
            self.model.trainer.policy.valid_extended_batch = False
            batch_save_path = os.path.join(save_path, "batch.npz")
            with open(batch_save_path, 'wb') as f:
                np.savez(f, self.model.trainer.policy.extended_batch)

    def load(self, load_path):
        """
        Load model state from checkpoint.

        Parameters
        ----------
        load_path : str
            Checkpoint directory to load model state.
        """

        # Load the TensorFlow graph
        if self.model.sess is None:
            self.model.setup()
        tf_load_path = os.path.join(load_path, "tf")
        self.saver.restore(self.model.sess, tf_load_path)

        # Load the Trainer
        # print("Loading Trainer...")
        trainer_load_path = os.path.join(load_path, "trainer.json")
        self.model.trainer.load(trainer_load_path)

        # Load the priority queue, if applicable
        # TBD: This should be in self.model.trainer.load or self.model.trainer.policy_optimizer.load after refactoring PolicyOptimizers to handle their own bookkeeping
        if self.model.trainer.priority_queue is not None:
            priority_queue_load_path = os.path.join(load_path, "priority_queue.npz")
            self.model.trainer.priority_queue.load(priority_queue_load_path)

        # Load the cache
        # print("Loading cache...")
        cache_load_path = os.path.join(load_path, "cache.csv")
        cache_df = pd.read_csv(cache_load_path)
        cache_df["tokens"] = cache_df["tokens"].str.split(",")
        programs = [from_tokens(np.array(tokens, dtype=np.int32)) for tokens in cache_df["tokens"]]
        for p, r in zip(programs, cache_df["rewards"]):
            p.r = r

        # Load the extra samples
        batch_save_path = os.path.join(load_path, "batch.npz")
        if os.path.isfile(batch_save_path):
            npzfile = np.load(batch_save_path, allow_pickle=True)
            self.model.trainer.policy.extended_batch = npzfile['arr_0']
            self.model.trainer.policy.valid_extended_batch = True
