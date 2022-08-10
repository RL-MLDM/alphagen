from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from sb3_contrib.common.maskable.distributions import make_masked_proba_distribution, MaskableDistribution
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from torch import nn


class MRActorCriticPolicy(ActorCriticPolicy):
    """
    Recurrent policy class for actor-critic algorithms (has both policy and value prediction).
    To be used with A2C, PPO and the likes.
    It assumes that both the actor and the critic LSTM
    have the same architecture.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param lstm_hidden_size: Number of hidden units for each LSTM layer.
    :param n_lstm_layers: Number of LSTM layers.
    :param shared_lstm: Whether the LSTM is shared between the actor and the critic
        (in that case, only the actor gradient is used)
        By default, the actor and the critic have two separate LSTM.
    :param enable_critic_lstm: Use a seperate LSTM for the critic.
    :param lstm_kwargs: Additional keyword arguments to pass the the LSTM
        constructor.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 256,
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.lstm_output_dim = lstm_hidden_size
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        self.lstm_kwargs = lstm_kwargs or {}
        self.shared_lstm = shared_lstm
        self.enable_critic_lstm = enable_critic_lstm
        self.lstm_actor = nn.LSTM(
            self.features_dim,
            lstm_hidden_size,
            num_layers=n_lstm_layers,
            **self.lstm_kwargs,
        )
        # For the predict() method, to initialize hidden states
        # (n_lstm_layers, batch_size, lstm_hidden_size)
        self.lstm_hidden_state_shape = (n_lstm_layers, 1, lstm_hidden_size)
        self.critic = None
        self.lstm_critic = None
        assert not (
            self.shared_lstm and self.enable_critic_lstm
        ), "You must choose between shared LSTM, separate or no LSTM for the critic"

        # No LSTM for the critic, we still need to convert
        # output of features extractor to the correct size
        # (size of the output of the actor lstm)
        if not (self.shared_lstm or self.enable_critic_lstm):
            self.critic = nn.Linear(self.features_dim, lstm_hidden_size)

        # Use a separate LSTM for the critic
        if self.enable_critic_lstm:
            self.lstm_critic = nn.LSTM(
                self.features_dim,
                lstm_hidden_size,
                num_layers=n_lstm_layers,
                **self.lstm_kwargs,
            )

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        self.action_dist = make_masked_proba_distribution(action_space)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.mlp_extractor = MlpExtractor(
            self.lstm_output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    @staticmethod
    def _process_sequence(
        features: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        lstm: nn.LSTM,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Do a forward pass in the LSTM network.

        :param features: Input tensor
        :param lstm_states: previous cell and hidden states of the LSTM
        :param episode_starts: Indicates when a new episode starts,
            in that case, we need to reset LSTM states.
        :param lstm: LSTM object.
        :return: LSTM output and updated LSTM states.
        """
        # LSTM logic
        # (sequence length, batch size, features dim)
        # (batch size = n_envs for data collection or n_seq when doing gradient update)
        n_seq = lstm_states[0].shape[1]
        # Batch to sequence
        # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
        # note: max length (max sequence length) is always 1 during data collection
        features_sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # If we don't have to reset the state in the middle of a sequence
        # we can avoid the for loop, which speeds up things
        if th.all(episode_starts == 0.0):
            lstm_output, lstm_states = lstm(features_sequence, lstm_states)
            lstm_output = th.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
            return lstm_output, lstm_states

        lstm_output = []
        # Iterate over the sequence
        for features, episode_start in zip_strict(features_sequence, episode_starts):
            hidden, lstm_states = lstm(
                features.unsqueeze(dim=0),
                (
                    # Reset the states at the beginning of a new episode
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
                ),
            )
            lstm_output += [hidden]
        # Sequence to batch
        # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
        lstm_output = th.flatten(th.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)
        return lstm_output, lstm_states

    def forward(
        self,
        obs: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, RNNStates]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation. Observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        # latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, lstm_states_pi = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        """
        Retrieve action distribution given the latent codes.
        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def get_distribution(
        self,
        obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[Distribution, Tuple[th.Tensor, ...]]:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the action distribution and new hidden states.
        """
        features = self.extract_features(obs)
        latent_pi, lstm_states = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution, lstm_states

    def predict_values(
        self,
        obs: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(features, lstm_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Use LSTM from the actor
            latent_pi, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        lstm_states: RNNStates,
        episode_starts: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation.
        :param actions:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, _ = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _predict(
        self,
        observation: th.Tensor,
        lstm_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, ...]]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy and hidden states of the RNN
        """
        distribution, lstm_states = self.get_distribution(
            observation, lstm_states,
            episode_starts, action_masks
        )
        return distribution.get_actions(deterministic=deterministic), lstm_states

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param state:
        :param episode_start:
        :param action_masks:
        :param observation: the input observation
        :param lstm_states: The last hidden and memory states for the LSTM.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the lstm states in that case).
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[list(observation.keys())[0]].shape[0]
        else:
            n_envs = observation.shape[0]
        # state : (n_layers, n_envs, dim)
        if state is None:
            # Initialize hidden states to zeros
            state = np.concatenate([np.zeros(self.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1)
            state = (state, state)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            # Convert to PyTorch tensors
            states = th.tensor(state[0]).float().to(self.device), th.tensor(state[1]).float().to(self.device)
            episode_starts = th.tensor(episode_start).float().to(self.device)
            actions, states = self._predict(
                observation,
                lstm_states=states,
                episode_starts=episode_starts,
                deterministic=deterministic,
                action_masks=action_masks,
            )
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())

        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions[0]

        return actions, states