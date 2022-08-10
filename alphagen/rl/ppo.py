from typing import Dict, Optional, List, Tuple
import time
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

from torch.optim import Adam
from torch.nn.utils.clip_grad import clip_grad_value_

from alphagen.data.tokens import Token
from alphagen.rl.env.core import AlphaEnvCore
from alphagen.rl.policy import Policy


class _Recorder:
    @dataclass
    class _Record:
        count: int = 0
        max: float = -np.inf
        min: float = np.inf
        sum: float = 0.

        def update(self, value: float) -> None:
            value = float(value)
            self.count += 1
            self.max = max(self.max, value)
            self.min = min(self.min, value)
            self.sum += value

    def __init__(self) -> None:
        self._state: Dict[str, _Recorder._Record] = {}

    def add_record(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if k not in self._state:
                self._state[k] = _Recorder._Record()
            self._state[k].update(v)

    def reset(self) -> None:
        self._state = {}

    def log_stats(self) -> None:
        max_k_width = max(len(k) for k in self._state.keys())
        for k, v in self._state.items():
            if v.count == 1:
                print(f"{k:{max_k_width}}: {v.sum:10.4f}")
            else:
                print(f"{k:{max_k_width}}: mean = {v.sum / v.count:10.4f}, "
                      f"range = [{v.min:10.4f}, {v.max:10.4f}], "
                      f"total = {v.count}")


@dataclass
class _EpochData:
    observations: List[Tuple[List[Token], dict]]
    actions: List[Token]
    returns: np.ndarray
    advantages: np.ndarray
    log_probs: np.ndarray


class _Buffer:
    """PPO buffer implementation adapted from OpenAI Spinning Up"""

    def __init__(self, size: int, gamma: float = 0.99, lambda_: float = 0.95):
        self._observations: List[Tuple[List[Token], dict]] = [([], {}) for _ in range(size)]
        self._actions = [Token() for _ in range(size)]
        self._advantages = np.zeros(size, dtype=np.float32)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._returns = np.zeros(size, dtype=np.float32)
        self._values = np.zeros(size, dtype=np.float32)
        self._log_probs = np.zeros(size, dtype=np.float32)
        self._gamma, self._lambda = gamma, lambda_
        self._ptr, self._path_start_idx, self._size = 0, 0, size

    def store(self, observation: Tuple[List[Token], dict],
              action: Token, reward: float, value: float, log_prob: float):
        assert self._ptr < self._size
        self._observations[self._ptr] = observation
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._values[self._ptr] = value
        self._log_probs[self._ptr] = log_prob
        self._ptr += 1

    def finish_path(self, last_val: float = 0):
        def _discounted_cumsum(arr: np.ndarray, factor: float) -> np.ndarray:
            arange = np.arange(arr.shape[0])
            tril = np.tril(factor ** (arange[:, None] - arange))
            return tril @ arr

        path_slice = slice(self._path_start_idx, self._ptr)
        rews = np.append(self._rewards[path_slice], last_val)
        vals = np.append(self._values[path_slice], last_val)

        deltas = rews[:-1] + self._gamma * vals[1:] - vals[:-1]
        self._advantages[path_slice] = _discounted_cumsum(deltas, self._gamma * self._lambda)
        self._returns[path_slice] = _discounted_cumsum(rews, self._gamma)[:-1]

        self._path_start_idx = self._ptr

    def get(self) -> _EpochData:
        def _normalize(arr: np.ndarray) -> np.ndarray:
            mean, std = arr.mean(), arr.std()
            return (arr - mean) / std

        assert self._ptr == self._size    # buffer has to be full before you can get
        self._ptr, self._path_start_idx = 0, 0
        return _EpochData(
            self._observations, self._actions,
            # self._returns, _normalize(self._advantages),
            self._returns, self._advantages,
            self._log_probs
        )


def ppo(env: AlphaEnvCore, policy: Policy, seed: Optional[int] = None,
        steps_per_epoch: int = 4000, epochs: int = 50,
        gamma: float = 0.99, clip_ratio: float = 0.2,
        entropy_weight: Optional[float] = None,
        pi_lr: float = 3e-4, vf_lr: float = 1e-3,
        train_pi_iters: int = 80, train_v_iters: int = 80,
        lambda_: float = 0.97, max_ep_len: int = 1000, save_freq: int = 10):
    env.reset(seed=seed)
    buf = _Buffer(steps_per_epoch, gamma, lambda_)
    rec = _Recorder()

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data: _EpochData) -> Tensor:
        obs, act, adv, logp_old = data.observations, data.actions, data.advantages, data.log_probs
        logp_old = torch.tensor(logp_old, device=policy.device)
        adv = torch.tensor(adv, device=policy.device)

        # Policy loss
        log_probs = []
        entropies = []
        for i, o in enumerate(obs):
            _, logp, entropy = policy.get_action(*o, act[i])
            log_probs.append(logp)
            entropies.append(entropy)
        logp = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        target = torch.min(ratio * adv, clip_adv)
        if entropy_weight is not None:
            target = target + entropy_weight * entropies

        loss_pi = -target.mean()
        return loss_pi

    # Set up function for computing value loss
    def compute_loss_v(data: _EpochData) -> Tensor:
        obs, ret = data.observations, data.returns
        ret = torch.tensor(ret, device=policy.device)

        values = [policy.get_value(*o) for o in obs]
        values = torch.stack(values)

        return ((values - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(policy.policy_parameters(), lr=pi_lr)
    vf_optimizer = Adam(policy.value_parameters(), lr=vf_lr)

    def update():
        data = buf.get()

        pi_l_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for _ in tqdm(range(train_pi_iters), desc="Policy Net"):
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            clip_grad_value_(policy.policy_parameters(), 1.0)
            pi_optimizer.step()

        # Value function learning
        for _ in tqdm(range(train_v_iters), desc="Value  Net"):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            clip_grad_value_(policy.value_parameters(), 1.0)
            vf_optimizer.step()

        # Log changes from update
        rec.add_record(
            LossPi=pi_l_old, LossV=v_l_old,
            DeltaLossPi=(loss_pi.item() - pi_l_old),    # type: ignore
            DeltaLossV=(loss_v.item() - v_l_old)        # type: ignore
        )

    # Prepare for interaction with environment
    start_time = time.time()
    obs, eps_ret, eps_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        print(f"[Epoch {epoch}]")

        for t in tqdm(range(steps_per_epoch), desc="Simulation"):
            action, logp, entropy, value = policy.get_action_value(*obs)
            next_obs, reward, done, info = env.step(action)
            eps_ret += reward
            eps_len += 1

            # save and log
            buf.store(obs, action, reward, value.item(), logp.item())
            rec.add_record(VVals=value, Entropy=entropy.item())

            # Update obs (critical!)
            obs = next_obs, info

            timeout = eps_len == max_ep_len
            terminal = done or timeout
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    action, logp, entropy, value = policy.get_action_value(*obs)
                else:
                    value = 0
                buf.finish_path(float(value))
                if terminal:
                    rec.add_record(EpRet=eps_ret, EpLen=eps_len)
                obs, eps_ret, eps_len = env.reset(), 0, 0

        # Perform PPO update!
        update()

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            policy.save(f"out/policy-epoch-{epoch}.pkl")

        # Log info about epoch
        rec.add_record(TotalTime=time.time() - start_time)
        rec.log_stats()
        rec.reset()


if __name__ == "__main__":
    from alphagen.utils.random import reseed_everything
    from alphagen.data.expression import *

    reseed_everything(0)

    device = torch.device("cuda:0")
    env = AlphaEnvCore("csi100", "2018-01-01", "2018-12-31", device)
    policy = Policy(
        n_encoder_layers=4,
        d_model=256,
        n_head=8,
        d_ffn=1024,
        dropout=0.1,
        operators=[Add, Sub, Mul, Div, Ref, Abs, EMA, Mean, Std],
        delta_time_range=(1, 31),
        device=device
    )
    ppo(
        env, policy, seed=0,
        steps_per_epoch=500, epochs=200,
        pi_lr=5e-6, vf_lr=5e-6, entropy_weight=0.01,
        train_pi_iters=10, train_v_iters=10,
        save_freq=4
    )
