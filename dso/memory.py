"""Classes for memory buffers, priority queues, and quantile estimation."""

import heapq
from collections import namedtuple

import numpy as np

from dso.program import from_tokens
from dso.utils import pad_action_obs_priors


Batch = namedtuple(
    "Batch", ["actions", "obs", "priors", "lengths", "rewards", "on_policy"])


# TBD: This should be member function of Batch class
def save_batch(B, save_path):
    """Save Batch to file."""

    with open(save_path, "wb") as f:
        np.savez(f, **dict(B._asdict()))


# TBD: This should be class function of Batch class
def load_batch(save_path):
    """Load Batch from file."""

    B = np.load(save_path)
    B = Batch(**B)
    return B


def make_queue(policy=None, priority=False, capacity=np.inf, seed=0):
    """Factory function for various Queues.

    Parameters
    ----------
    policy : dso.policy.Policy
        Reference to the Policy, used to compute probabilities of items in
        the Queue.

    priority : bool
        If True, returns an object inheriting UniquePriorityQueue. Otherwise,
        returns an object inheriting from UniqueQueue.

    capacity : int
        Maximum queue length.

    seed : int
        RNG seed used for random sampling.

    Returns
    -------
    queue : ProgramQueue
        Dynamic class inheriting from ProgramQueueMixin and a Queue subclass.
    """

    if priority:
        Base = UniquePriorityQueue
    else:
        Base = UniqueQueue

    class ProgramQueue(ProgramQueueMixin, Base):
        def __init__(self, policy, capacity, seed):
            ProgramQueueMixin.__init__(self, policy)
            Base.__init__(self, capacity, seed)

    queue = ProgramQueue(policy, capacity, seed)
    return queue


def get_samples(batch, key):
    """
    Returns a sub-Batch with samples from the given indices.

    Parameters
    ----------
    key : int or slice
        Indices of samples to return.

    Returns
    -------
    batch : Batch
        Sub-Batch with samples from the given indices.
    """

    batch = Batch(
        actions=batch.actions[key],
        obs=batch.obs[key],
        priors=batch.priors[key],
        lengths=batch.lengths[key],
        rewards=batch.rewards[key],
        on_policy=batch.on_policy[key])
    return batch


# Adapted from https://github.com/tensorflow/models/blob/1af55e018eebce03fb61bba9959a04672536107d/research/brain_coder/common/utils.py
class ItemContainer(object):
    """Class for holding an item with its score.

    Defines a comparison function for use in the heap-queue.
    """

    def __init__(self, score, item, extra_data):
        self.item = item
        self.score = score
        self.extra_data = extra_data

    def __lt__(self, other):
        assert isinstance(other, type(self))
        return self.score < other.score

    def __eq__(self, other):
        assert isinstance(other, type(self))
        return self.item == other.item

    def __iter__(self):
        """Allows unpacking like a tuple."""
        yield self.score
        yield self.item
        yield self.extra_data

    def __repr__(self):
        """String representation of this item.

        `extra_data` is not included in the representation. We are assuming that
        `extra_data` is not easily interpreted by a human (if it was, it should be
        hashable, like a string or tuple).

        Returns:
            String representation of `self`.
        """
        return str((self.score, self.item))

    def __str__(self):
        return repr(self)


class Queue(object):
    """Abstract class for queue that must define a push and pop routine"""

    def __init__(self, capacity, seed=0):
        self.capacity = capacity
        self.rng = np.random.RandomState(seed)
        self.heap = []
        self.unique_items = set()

    def push(self, score, item, extra_data):
        raise NotImplementedError

    def pop(self):
        raise NotImplementedError

    def random_sample(self, sample_size):
        """Uniform randomly select items from the queue.

        Args:
            sample_size: Number of random samples to draw. The same item can be
                    sampled multiple times.

        Returns:
            List of sampled items (of length `sample_size`). Each element in the list
            is a tuple: (item, extra_data).
        """
        idx = self.rng.choice(len(self.heap), sample_size, )
        return [(self.heap[i].item, self.heap[i].extra_data) for i in idx]

    def __len__(self):
        return len(self.heap)

    def __iter__(self):
        for _, item, _ in self.heap:
            yield item

    def __repr__(self):
        return '[' + ', '.join(repr(c) for c in self.heap) + ']'

    def __str__(self):
        return repr(self)


class UniqueQueue(Queue):
    """A queue in which duplicates are not allowed. Instead, adding a duplicate
    moves that item to the back of the queue."""

    def push(self, score, item, extra_data=None):
        """Push an item onto the queue, or move it to the back if already
        present.

        Score is unused but included as an argument to follow the interface.
        """

        container = ItemContainer(None, item, extra_data)

        # If the item is already in the queue, move it to the back of the queue
        # and return
        if item in self.unique_items:
            self.heap.remove(container)
            self.heap.append(container)
            return

        # If the queue is at capacity, first pop the front of the queue
        if len(self.heap) >= self.capacity:
            self.pop()

        # Add the item
        self.heap.append(container)
        self.unique_items.add(item)

    def pop(self):
        """Pop the front of the queue (the oldest item)."""

        if not self.heap:
            return ()
        score, item, extra_data = self.heap.pop(0)
        self.unique_items.remove(item)
        return (score, item, extra_data)


# Adapted from https://github.com/tensorflow/models/blob/1af55e018eebce03fb61bba9959a04672536107d/research/brain_coder/common/utils.py
class UniquePriorityQueue(Queue):
    """A priority queue where duplicates are not added.

    The top items by score remain in the queue. When the capacity is reached,
    the lowest scored item in the queue will be dropped.
    """

    def push(self, score, item, extra_data=None):
        """Push an item onto the queue.

        If the queue is at capacity, the item with the smallest score will be
        dropped. Note that it is assumed each item has exactly one score. The same
        item with a different score will still be dropped.

        Args:
            score: Number used to prioritize items in the queue. Largest scores are
                    kept in the queue.
            item: A hashable item to be stored. Duplicates of this item will not be
                    added to the queue.
            extra_data: An extra (possible not hashable) data to store with the item.
        """
        if item in self.unique_items:
            return
        if len(self.heap) >= self.capacity:
            _, popped_item, _ = heapq.heappushpop(
                self.heap, ItemContainer(score, item, extra_data))
            self.unique_items.add(item)
            self.unique_items.remove(popped_item)
        else:
            heapq.heappush(self.heap, ItemContainer(score, item, extra_data))
            self.unique_items.add(item)

    def pop(self):
        """Pop the item with the lowest score.

        Returns:
            score: Item's score.
            item: The item that was popped.
            extra_data: Any extra data stored with the item.
        """
        if not self.heap:
            return ()
        score, item, extra_data = heapq.heappop(self.heap)
        self.unique_items.remove(item)
        return score, item, extra_data

    def get_max(self):
        """Peek at the item with the highest score.

        Returns:
            Same as `pop`.
        """
        if not self.heap:
            return ()
        score, item, extra_data = heapq.nlargest(1, self.heap)[0]
        return score, item, extra_data

    def get_min(self):
        """Peek at the item with the lowest score.

        Returns:
            Same as `pop`.
        """
        if not self.heap:
            return ()
        score, item, extra_data = heapq.nsmallest(1, self.heap)[0]
        return score, item, extra_data

    def iter_in_order(self):
        """Iterate over items in the queue from largest score to smallest.

        Yields:
            item: Hashable item.
            extra_data: Extra data stored with the item.
        """
        for _, item, extra_data in heapq.nlargest(len(self.heap), self.heap):
            yield item, extra_data


class ProgramQueueMixin():
    """A mixin for Queues with additional utilities specific to Batch and
    Program."""

    def __init__(self, policy=None):
        self.policy = policy

    def push_sample(self, sample, program):
        """
        Push a single sample corresponding to Program to the queue.

        Parameters
        ----------
        sample : Batch
            A Batch comprising a single sample.

        program : Program
            Program corresponding to the sample.
        """

        id_ = program.str
        score = sample.rewards
        self.push(score, id_, sample)

    def push_batch(self, batch, programs):
        """Push a Batch corresponding to Programs to the queue."""

        for i, program in enumerate(programs):
            sample = get_samples(batch, i)
            self.push_sample(sample, program)

    def push_best(self, batch, programs):
        """Push the single best sample from a Batch"""

        i = np.argmax(batch.rewards)
        sample = get_samples(batch, i)
        program = programs[i]
        self.push_sample(sample, program)

    def sample_batch(self, sample_size):
        """Randomly select items from the queue and return them as a Batch."""

        assert len(self.heap) > 0, "Cannot sample from an empty queue."
        samples = [sample for (id_, sample) in self.random_sample(sample_size)]
        batch = self._make_batch(samples)
        return batch

    def _make_batch(self, samples):
        """Turns a list of samples into a Batch."""

        # Pad up to max length across samples
        max_len = max([s.actions.shape[0] for s in samples])
        padded_aop = [pad_action_obs_priors(np.expand_dims(s.actions, axis=0),
                                            np.expand_dims(s.obs, axis=0),
                                            np.expand_dims(s.priors, axis=0),
                                            pad_length=max_len - s.actions.shape[0]) for s in samples]

        actions, obs, priors = zip(*padded_aop)
        actions = np.concatenate(actions, axis=0)
        obs = np.concatenate(obs, axis=0)
        priors = np.concatenate(priors, axis=0)
        lengths = np.array([s.lengths for s in samples], dtype=np.int32)
        rewards = np.array([s.rewards for s in samples], dtype=np.float32)
        on_policy = np.array([s.on_policy for s in samples], dtype=np.bool)
        batch = Batch(actions=actions, obs=obs, priors=priors,
                      lengths=lengths, rewards=rewards, on_policy=on_policy)
        return batch

    def to_batch(self):
        """Return the entire queue as a Batch."""

        samples = [container.extra_data for container in self.heap]
        batch = self._make_batch(samples)
        return batch

    def compute_probs(self):
        """Computes the probabilities of items in the queue according to the
        Policy."""

        if self.policy is None:
            raise RuntimeError("Cannot compute probabilities. This Queue does \
                not have a Policy.")
        return self.policy.compute_probs(self.to_batch())

    def get_rewards(self):
        """Returns the rewards"""

        r = [container.extra_data.rewards for container in self.heap]
        return r

    def save(self, save_path):
        """Save the contents of the queue to file."""

        B = self.to_batch()
        save_batch(B, save_path)

    def load(self, load_path):
        """Load the contents of the queue from file."""

        B = load_batch(load_path)
        programs = [from_tokens(np.array(tokens, dtype=np.int32)) for tokens in B.actions]
        for p, r in zip(programs, B.rewards):
            p.r = r
        self.push_batch(B, programs)
