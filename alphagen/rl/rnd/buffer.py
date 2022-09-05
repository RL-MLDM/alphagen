from typing import Generator, NamedTuple, Optional, Tuple, Union, cast

import numpy as np
import torch as th

from gym import spaces
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


class MultiheadBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor


class MultiheadRolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_heads: int = 1,
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(MultiheadRolloutBuffer, self).__init__(
            buffer_size,
            observation_space, action_space,
            device, n_envs=n_envs
        )
        self.n_heads = n_heads
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.reset()

    @property
    def mask_dims(self) -> int:
        if isinstance(self.action_space, spaces.Discrete):
            return self.action_space.n              # type: ignore
        if isinstance(self.action_space, spaces.MultiDiscrete):
            return sum(self.action_space.nvec)      # type: ignore
        if isinstance(self.action_space, spaces.MultiBinary):
            return 2 * self.action_space.n          # type: ignore
        raise ValueError(f"Unsupported action space {type(self.action_space)}")

    def reset(self) -> None:
        def zero_array(*extra_dims: int) -> np.ndarray:
            dims = [self.buffer_size, self.n_envs, *extra_dims]
            return np.zeros(dims, dtype=np.float32)

        self.observations = zero_array(*cast(Tuple[int, ...], self.obs_shape))
        self.actions = zero_array(self.action_dim)
        self.rewards = zero_array(self.n_heads)
        self.returns = zero_array(self.n_heads)
        self.episode_starts = zero_array()
        self.values = zero_array(self.n_heads)
        self.log_probs = zero_array()
        self.advantages = zero_array(self.n_heads)
        self.action_masks = zero_array(self.mask_dims)
        self.generator_ready = False

        super(MultiheadRolloutBuffer, self).reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        vals: np.ndarray = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = vals
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            assert isinstance(self.obs_shape, tuple)
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))   # type: ignore

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[MultiheadBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        # Prepare the data
        if not self.generator_ready:
            self._prepare_data()

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _prepare_data(self) -> None:
        _tensor_names = [
            "observations",
            "actions", "log_probs",
            "values", "advantages", "returns",
            "action_masks"
        ]
        for tensor in _tensor_names:
            self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
        self.generator_ready = True

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None
    ) -> MultiheadBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds],
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds],
            self.returns[batch_inds],
            self.action_masks[batch_inds].reshape(-1, self.mask_dims)
        )
        return MultiheadBufferSamples(*tuple(map(self.to_torch, data)))
