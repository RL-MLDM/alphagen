"""Core deep symbolic optimizer construct."""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import zlib
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import random
from time import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import commentjson as json

from dso.task import set_task
from dso.train import Trainer
from dso.checkpoint import Checkpoint
from dso.train_stats import StatsLogger
from dso.prior import make_prior
from dso.program import Program
from dso.config import load_config
from dso.tf_state_manager import make_state_manager

from dso.policy.policy import make_policy
from dso.policy_optimizer import make_policy_optimizer

class DeepSymbolicOptimizer():
    """
    Deep symbolic optimization model. Includes model hyperparameters and
    training configuration.

    Parameters
    ----------
    config : dict or str
        Config dictionary or path to JSON.

    Attributes
    ----------
    config : dict
        Configuration parameters for training.

    Methods
    -------
    train
        Builds and trains the model according to config.
    """

    def __init__(self, config=None):
        self.set_config(config)
        self.sess = None

    def setup(self):

        # Clear the cache and reset the compute graph
        Program.clear_cache()
        tf.reset_default_graph()

        # Generate objects needed for training and set seeds
        self.pool = self.make_pool_and_set_task()
        self.set_seeds() # Must be called _after_ resetting graph and _after_ setting task

        # Limit TF to single thread to prevent "resource not available" errors in parallelized runs
        session_config = tf.ConfigProto(intra_op_parallelism_threads=1,
                                        inter_op_parallelism_threads=1)
        self.sess = tf.Session(config=session_config)

        # Setup logdirs and output files
        self.output_file = self.make_output_file()
        self.save_config()

        # Prepare training parameters
        self.prior = self.make_prior()
        self.state_manager = self.make_state_manager()
        self.policy = self.make_policy()
        self.policy_optimizer = self.make_policy_optimizer()
        self.gp_controller = self.make_gp_controller()
        self.logger = self.make_logger()
        self.trainer = self.make_trainer()
        self.checkpoint = self.make_checkpoint()

    def train_one_step(self, override=None):
        """
        Train one iteration.
        """

        # Setup the model
        if self.sess is None:
            self.setup()

        # Run one step
        assert not self.trainer.done, "Training has already completed!"
        self.trainer.run_one_step(override)
        
        # Maybe save next checkpoint
        self.checkpoint.update()

        # If complete, return summary
        if self.trainer.done:
            return self.finish()

    def train(self):
        """
        Train the model until completion.
        """

        # Setup the model
        self.setup()

        # Train the model until done
        while not self.trainer.done:
            result = self.train_one_step()

        return result

    def finish(self):
        """
        After training completes, finish up and return summary dict.
        """

        # Return statistics of best Program
        p = self.trainer.p_r_best
        result = {"seed" : self.config_experiment["seed"]} # Seed listed first
        result.update({"r" : p.r})
        result.update(p.evaluate)
        result.update({
            "expression" : repr(p.sympy_expr),
            "traversal" : repr(p),
            "program" : p
        })

        # Save all results available only after all iterations are finished. Also return metrics to be added to the summary file
        results_add = self.logger.save_results(self.pool, self.trainer.nevals)
        result.update(results_add)

        # Close the pool
        if self.pool is not None:
            self.pool.close()

        return result

    def set_config(self, config):
        config = load_config(config)

        self.config = defaultdict(dict, config)
        self.config_task = self.config["task"]
        self.config_prior = self.config["prior"]
        self.config_logger = self.config["logging"]
        self.config_training = self.config["training"]
        self.config_state_manager = self.config["state_manager"]
        self.config_policy = self.config["policy"]
        self.config_policy_optimizer = self.config["policy_optimizer"]
        self.config_gp_meld = self.config["gp_meld"]
        self.config_experiment = self.config["experiment"]
        self.config_checkpoint = self.config["checkpoint"]

    def save_config(self):
        # Save the config file
        if self.output_file is not None:
            path = os.path.join(self.config_experiment["save_path"],
                                "config.json")
            # With run.py, config.json may already exist. To avoid race
            # conditions, only record the starting seed. Use a backup seed
            # in case this worker's seed differs.
            backup_seed = self.config_experiment["seed"]
            if not os.path.exists(path):
                if "starting_seed" in self.config_experiment:
                    self.config_experiment["seed"] = self.config_experiment["starting_seed"]
                    del self.config_experiment["starting_seed"]
                with open(path, 'w') as f:
                    json.dump(self.config, f, indent=3)
            self.config_experiment["seed"] = backup_seed

    def set_seeds(self):
        """
        Set the tensorflow, numpy, and random module seeds based on the seed
        specified in config. If there is no seed or it is None, a time-based
        seed is used instead and is written to config.
        """

        seed = self.config_experiment.get("seed")

        # Default uses current time in milliseconds, modulo 1e9
        if seed is None:
            seed = round(time() * 1000) % int(1e9)
            self.config_experiment["seed"] = seed

        # Shift the seed based on task name
        # This ensures a specified seed doesn't have similarities across different task names
        task_name = Program.task.name
        shifted_seed = seed + zlib.adler32(task_name.encode("utf-8"))

        # Set the seeds using the shifted seed
        tf.set_random_seed(shifted_seed)
        np.random.seed(shifted_seed)
        random.seed(shifted_seed)

    def make_prior(self):
        prior = make_prior(Program.library, self.config_prior)
        return prior

    def make_state_manager(self):
        state_manager = make_state_manager(self.config_state_manager)
        return state_manager

    def make_trainer(self):
        trainer = Trainer(self.sess,
                          self.policy,
                          self.policy_optimizer,
                          self.gp_controller,
                          self.logger,
                          self.pool,
                          **self.config_training)
        return trainer

    def make_logger(self):
        logger = StatsLogger(self.sess,
                             self.output_file,
                             **self.config_logger)
        return logger

    def make_checkpoint(self):
        checkpoint = Checkpoint(self,
                                **self.config_checkpoint)
        return checkpoint

    def make_policy_optimizer(self):
        policy_optimizer = make_policy_optimizer(self.sess,
                                                 self.policy,
                                                 **self.config_policy_optimizer)
        return policy_optimizer

    def make_policy(self):
        policy = make_policy(self.sess,
                             self.prior,
                             self.state_manager,
                             **self.config_policy)
        return policy

    def make_gp_controller(self):
        if self.config_gp_meld.pop("run_gp_meld", False):
            from dso.gp.gp_controller import GPController
            gp_controller = GPController(self.prior,
                                         self.config_prior,
                                         **self.config_gp_meld)
        else:
            gp_controller = None
        return gp_controller

    def make_pool_and_set_task(self):
        # Create the pool and set the Task for each worker

        # Set complexity and const optimizer here so pool can access them
        # Set the complexity function
        complexity = self.config_training["complexity"]
        Program.set_complexity(complexity)

        # Set the constant optimizer
        const_optimizer = self.config_training["const_optimizer"]
        const_params = self.config_training["const_params"]
        const_params = const_params if const_params is not None else {}
        Program.set_const_optimizer(const_optimizer, **const_params)

        pool = None
        n_cores_batch = self.config_training.get("n_cores_batch")
        if n_cores_batch is not None:
            if n_cores_batch == -1:
                n_cores_batch = cpu_count()
            if n_cores_batch > 1:
                pool = Pool(n_cores_batch,
                            initializer=set_task,
                            initargs=(self.config_task,))

        # Set the Task for the parent process
        set_task(self.config_task)

        return pool

    def make_output_file(self):
        """Generates an output filename"""

        # If logdir is not provided (e.g. for pytest), results are not saved
        if self.config_experiment.get("logdir") is None:
            self.save_path = None
            print("WARNING: logdir not provided. Results will not be saved to file.")
            return None

        # When using run.py, timestamp is already generated
        timestamp = self.config_experiment.get("timestamp")
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            self.config_experiment["timestamp"] = timestamp

        # Generate save path
        task_name = Program.task.name
        if self.config_experiment["exp_name"] is None:
            save_path = os.path.join(
                self.config_experiment["logdir"],
                '_'.join([task_name, timestamp]))
        else:
            save_path = os.path.join(
                self.config_experiment["logdir"],
                self.config_experiment["exp_name"])

        self.config_experiment["task_name"] = task_name
        self.config_experiment["save_path"] = save_path
        os.makedirs(save_path, exist_ok=True)

        seed = self.config_experiment["seed"]
        output_file = os.path.join(save_path,
                                   "dso_{}_{}.csv".format(task_name, seed))

        self.save_path = save_path

        return output_file

    def save(self, save_path=None):
        self.checkpoint.save(save_path)

    def load(self, load_path):
        self.checkpoint.load(load_path)
