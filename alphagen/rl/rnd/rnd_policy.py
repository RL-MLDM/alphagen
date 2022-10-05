from functools import partial
from typing import Any, Dict, Optional, Tuple, Type, Union, List

import gym
from gym import spaces
import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor
)

from sb3_contrib.common.maskable.distributions import MaskableDistribution, make_masked_proba_distribution

from alphagen.rl.rnd.rnd_nets import RNDNet

from .mlp_extractor import MlpExtractor
from ...utils.pytorch_utils import MapperModule


class RNDPolicy(BasePolicy):
    """
    RND policy

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    `net_arch`: "pi" -> policy, "vf" -> values
    `rnd_arch`: "rn" -> rnd
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    `features_extractor: <>_class(**<>_kwargs)`
    `optimizer: <>_class(lr=lr, **<>_kwargs)`
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        rnd_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            net_arch = [{"pi": [64, 64], "vf": [64, 64]}]
        if rnd_arch is None:
            rnd_arch = [{"rn": [64, 64]}]

        self.net_arch = net_arch
        self.rnd_arch = rnd_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim
        self.action_dist = make_masked_proba_distribution(action_space)

        self._build(lr_schedule)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :param action_masks: Action masks to apply to the action distribution
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor.forward_only(features, ["pi", "vf"])
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                rnd_arch=self.rnd_arch,
                activation_fn=self.activation_fn,
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build_rnd(self) -> None:
        def get_rn(dct: Dict[str, th.Tensor]) -> th.Tensor: return dct["rn"]

        def factory() -> nn.Module:
            fe = self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
            mlp = MlpExtractor(self.features_dim, self.rnd_arch, self.activation_fn, self.device)
            mapper = MapperModule(get_rn)
            return nn.Sequential(fe, mlp, mapper)
        self.rnd = RNDNet(factory, self.device)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()
        self._build_rnd()

        latent_dim_pi = self.mlp_extractor.latent_dim("pi")
        latent_dim_vf = self.mlp_extractor.latent_dim("vf")
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        # 2 output heads: external/internal
        self.value_net = nn.Linear(latent_dim_vf, 2)

        if self.ortho_init:
            module_gains: Dict[nn.Module, float] = {            # type: ignore
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.rnd: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            module: nn.Module
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1), **self.optimizer_kwargs          # type: ignore
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        action_logits = self.action_net(latent_pi)              # type: ignore
        return self.action_dist.proba_distribution(action_logits=action_logits)     # type: ignore

    def _predict(
        self,
        observation: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :param action_masks: Action masks to apply to the action distribution
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation, action_masks).get_actions(deterministic=deterministic)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action and state from an observation (and optional state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :param action_masks: Action masks to apply to the action distribution
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        self.set_training_mode(False)

        obs, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(obs, deterministic=deterministic, action_masks=action_masks)
            # Convert to numpy
            actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)   # type: ignore

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions = actions[0]

        return actions, state

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        features = self.extract_features(obs)
        latent = self.mlp_extractor.forward(features)
        distribution = self._get_action_dist_from_latent(latent["pi"])
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent["vf"])
        entropy = distribution.entropy()
        assert entropy is not None
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor, action_masks: Optional[np.ndarray] = None) -> MaskableDistribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :param action_masks:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi, = self.mlp_extractor.forward_only(features, ["pi"])
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        latent_vf, = self.mlp_extractor.forward_only(features, ["vf"])
        return self.value_net(latent_vf)

    def predict_rnd(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]: return self.rnd(obs)
