import copy
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def build_mlp(
    input_dim: int,
    hidden_dims: Tuple[int, ...],
    output_dim: int,
    activation=nn.ReLU,
    output_activation=nn.Identity,
) -> nn.Sequential:
    """
    Build a standard multi-layer perceptron (MLP).

    What this function does:
    - This is just a helper function for quickly creating a feedforward network.
    - We use it to build the actor backbone and critic network.

    Why this exists:
    - In SAC, both the actor and critics are neural networks.
    - Their structures are usually simple MLPs.
    - Instead of manually writing many Linear + ReLU layers every time,
      we wrap that repeated pattern into one helper function.

    Input:
    - input_dim:
        Dimension of the input vector.
        Example: if state = [q1, q2, dq1, dq2], then input_dim = 4.

    - hidden_dims:
        Sizes of hidden layers.
        Example: (256, 256) means:
            Linear(input_dim -> 256)
            ReLU
            Linear(256 -> 256)
            ReLU

    - output_dim:
        Dimension of the final output vector.

    Output:
    - model:
        An nn.Sequential network.

    Example:
    - Actor backbone:
        state_dim -> hidden layers -> feature vector
    - Critic:
        (state_dim + action_dim) -> hidden layers -> scalar Q value
    """
    layers = []
    prev_dim = input_dim

    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(activation())
        prev_dim = h

    layers.append(nn.Linear(prev_dim, output_dim))
    layers.append(output_activation())

    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Policy network used by SAC.

    Big picture:
    - In reinforcement learning, the actor is the part that decides what action
      to take under the current state.
    - In SAC, the actor is stochastic, not deterministic.
    - That means the actor does NOT directly output one fixed action.
      Instead, it outputs the parameters of a Gaussian distribution:
          mean and log_std
      and then samples an action from that distribution.

    This class is responsible for:
    1) taking a state as input
    2) predicting Gaussian parameters for each action dimension
    3) sampling an action in a differentiable way
    4) squashing the action into [-1, 1] with tanh

    Why tanh is used:
    - The Gaussian sample can be any real number
    - But actions in control tasks are usually bounded
    - So we first sample in unconstrained space, then use tanh to map
      the action into a normalized bounded range

    Important:
    - This class only cares about dimensions.
    - It does not care whether the state means joint angles, end-effector
      position, velocities, or something else.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        ####################################################################
        # TODO:
        # Build the actor network.
        #
        # Idea:
        # - First use one shared MLP ("backbone") to extract features
        #   from the state.
        # - Then use two separate linear heads:
        #     1) mean_head:    predicts the Gaussian mean
        #     2) log_std_head: predicts the Gaussian log standard deviation
        #
        # Why two heads?
        # - In SAC, the policy is a Gaussian distribution.
        # - A Gaussian needs two parameters: mean and std.
        # - We usually predict log_std instead of std directly because it is
        #   numerically more stable, and later we can do:
        #       std = exp(log_std)
        #
        # Input:
        # - state: shape (B, state_dim)
        #
        # Output of backbone:
        # - features: shape (B, hidden_dims[-1])
        ####################################################################
        self.backbone = build_mlp(self.state_dim, hidden_dims[:-1], hidden_dims[-1])
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the Gaussian parameters of the policy.

        What this function does:
        - Given the current state, this function produces the distribution
          from which SAC will sample actions.
        - It does NOT sample the final action yet.
        - It only returns:
            mean     : where the Gaussian is centered
            log_std  : how spread out the Gaussian is

        Why this is important:
        - SAC needs a stochastic policy.
        - So before we can sample an action, we first need to define the
          Gaussian distribution for each action dimension.

        Input:
        - state: torch.Tensor of shape (B, state_dim)

        Output:
        - mean: torch.Tensor of shape (B, action_dim)
            Gaussian mean before tanh squashing

        - log_std: torch.Tensor of shape (B, action_dim)
            Log standard deviation before exponentiation

        Important note:
        - log_std should be restricted to a reasonable range.
        - If std becomes too large, actions become extremely noisy.
        - If std becomes too small, learning can become unstable.
        - So we clamp log_std into [log_std_min, log_std_max].
        """
        mean = None
        log_std = None

        ####################################################################
        # TODO:
        # Step 1) Pass state through the shared backbone to get features.
        # Step 2) Use mean_head to predict the Gaussian mean.
        # Step 3) Use log_std_head to predict the raw log standard deviation.
        # Step 4) Clamp log_std into a safe range.
        #
        # In short:
        #   state -> features -> (mean, log_std)
        ####################################################################
        mlp_head = self.backbone(state)
        mean = self.mean_head(mlp_head)
        log_std = self.log_std_head(mlp_head)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

        return mean, log_std

    def sample(
        self, state: torch.Tensor, eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        What this function does in SAC:
        - The actor outputs a Gaussian distribution, not a fixed action.
        - This function actually samples from that Gaussian and turns it into
          the final action used by SAC.

        Why this function is important:
        - During training, SAC wants the policy to remain stochastic.
        - But it also needs gradients to flow through the sampled action
          so the actor can be updated by gradient descent.
        - Therefore SAC uses the reparameterization trick.

        What "reparameterized sampling" means here:
        - Instead of treating sampling as an isolated random black box,
          we rewrite the sample as:
              pre_tanh_action = mean + std * noise
          where noise ~ N(0, 1)
        - This keeps randomness, but also allows gradients to flow back to
          mean and std.

        Why tanh is applied:
        - The Gaussian sample lives in (-inf, +inf)
        - But we usually want normalized actions in [-1, 1]
        - So:
              action = tanh(pre_tanh_action)

        Why log_prob is needed:
        - SAC does not only care about action quality (Q value),
          it also explicitly encourages policy entropy.
        - So the algorithm needs log_prob(action) when computing the
          actor loss and entropy temperature loss.

        Why there is a tanh correction term:
        - The Gaussian distribution is defined in pre-tanh space.
        - But the final action is after tanh.
        - Since tanh changes the density, the log-probability must be
          corrected using the change-of-variables term.

        Input:
        - state: torch.Tensor of shape (B, state_dim)

        Output:
        - action: torch.Tensor of shape (B, action_dim)
            Sampled action after tanh, roughly in [-1, 1]

        - log_prob: torch.Tensor of shape (B, 1)
            Log-probability of the sampled action, including tanh correction

        - mean_action: torch.Tensor of shape (B, action_dim)
            Deterministic action computed as tanh(mean)
            This is often used at test time

        Recommended mental picture:
        state
          -> actor forward()
          -> mean, log_std
          -> Normal(mean, std)
          -> rsample() to get pre_tanh_action
          -> tanh()
          -> final action
          -> compute corrected log_prob
        """
        action = None
        log_prob = None
        mean_action = None

        ####################################################################
        # TODO:
        # Implement SAC action sampling.
        #
        # Step 1) Get mean and log_std from forward().
        # Step 2) Convert log_std into std using exp().
        # Step 3) Create a Gaussian distribution Normal(mean, std).
        # Step 4) Use rsample(), not sample().
        #         Why?
        #         - rsample() supports the reparameterization trick
        #         - gradients can flow through the sampled action
        #
        # Step 5) Apply tanh to the pre_tanh_action to get the final action.
        #
        # Step 6) Compute the Gaussian log-probability of the PRE-TANH sample.
        #         Important:
        #         - the Gaussian is defined before tanh
        #         - so log_prob must start there
        #
        # Step 7) Subtract the tanh correction term:
        #             log(1 - action^2 + eps)
        #         This converts the density from pre-tanh space to
        #         post-tanh action space.
        #
        # Step 8) Sum log_prob across action dimensions and keep shape (B, 1).
        #
        # Step 9) Also compute mean_action = tanh(mean) for deterministic use.
        #
        # Shapes:
        # - pre_tanh_action: (B, action_dim)
        # - action:          (B, action_dim)
        # - log_prob:        (B, 1)
        # - mean_action:     (B, action_dim)
        ####################################################################
        mean, log_std = self.forward(state)
        
        std = torch.exp(log_std)
        distribution = Normal(mean, std)
        pre_tanh_action = distribution.rsample()
        action = torch.tanh(pre_tanh_action)
        
        log_prob = distribution.log_prob(pre_tanh_action)
        log_prob = log_prob - torch.log(1 - action.pow(2) + eps)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        mean_action = torch.tanh(mean)
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

        return action, log_prob, mean_action


class Critic(nn.Module):
    """
    Q-network used by SAC.

    Big picture:
    - The critic estimates how good an action is under a given state.
    - In other words, it approximates:
          Q(s, a)

    Intuition:
    - The actor proposes actions
    - The critic judges those actions
    - Then the actor learns to produce actions that the critic scores highly

    Why the critic takes both state and action:
    - A Q-function is not just about the state
    - It measures the value of taking a specific action at that state

    Output:
    - A single scalar per sample:
          q_value
      which estimates the expected long-term return
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        ####################################################################
        # TODO:
        # Build a critic MLP.
        #
        # Input to the critic:
        # - concatenated [state, action]
        # - shape: (B, state_dim + action_dim)
        #
        # Output:
        # - one scalar Q value per sample
        # - shape: (B, 1)
        #
        # Why concatenate state and action?
        # - Because Q(s, a) depends on both
        ####################################################################
        self.q_net = build_mlp(state_dim+action_dim, hidden_dims, 1)
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Q(s, a).

        What this function does:
        - It receives a state and an action
        - Then predicts how good that action is at that state

        Input:
        - state:  shape (B, state_dim)
        - action: shape (B, action_dim)

        Output:
        - q_value: shape (B, 1)

        Important:
        - We concatenate state and action along the last dimension before
          feeding them into the Q-network.
        """
        q_value = None

        ####################################################################
        # TODO:
        # Step 1) Concatenate state and action along dim=-1
        # Step 2) Feed the result into q_net
        ####################################################################
        x = torch.concatenate([state, action], dim=-1)
        q_value = self.q_net(x)
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

        return q_value


class ReplayBuffer:
    """
    Replay buffer for off-policy reinforcement learning.

    What this class does:
    - It stores past transitions collected from interacting with the environment.
    - During training, SAC randomly samples mini-batches from this buffer.

    Why this is needed:
    - SAC is an off-policy algorithm.
    - That means it does not only learn from the newest transition.
    - It reuses old experience many times, which improves sample efficiency.

    Stored transition:
    - state
    - action (unscaled)
    - reward
    - next_state
    - done

    Intuition:
    - Think of this as a memory bank of past experiences.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.capacity = capacity
        self.device = torch.device(device)

        self.ptr = 0
        self.size = 0

        ####################################################################
        # TODO:
        # Allocate numpy arrays for storing transitions.
        #
        # Suggested shapes:
        # - states:      (capacity, state_dim)
        # - actions:     (capacity, action_dim)
        # - rewards:     (capacity, 1)
        # - next_states: (capacity, state_dim)
        # - dones:       (capacity, 1)
        #
        # Why pre-allocate memory?
        # - It is faster and cleaner than growing Python lists forever.
        #
        # Why use a circular buffer?
        # - When the buffer is full, we overwrite the oldest data.
        ####################################################################
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)            # reward of the current step (first term in Bellman's equation)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)         
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        """
        Store one transition in the replay buffer.

        What this function does:
        - Write one transition at the current pointer position
        - Then move the pointer forward
        - If the buffer is full, wrap around and overwrite old data

        Input:
        - state:      shape (state_dim,)
        - action:     shape (action_dim,)
        - reward:     scalar
        - next_state: shape (state_dim,)
        - done:       scalar or bool

        Output:
        - None
        """
        ####################################################################
        # TODO:
        # Step 1) Write the transition into index self.ptr
        # Step 2) Move the circular pointer
        # Step 3) Update the current valid size
        ####################################################################
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity

        self.size = min(self.size + 1, self.capacity)
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch of past transitions.

        Why random sampling is used:
        - Consecutive environment transitions are strongly correlated
        - Neural networks train better when mini-batches are more i.i.d.-like
        - Random replay helps break temporal correlation

        Input:
        - batch_size: number of transitions to sample

        Output:
        - states:      shape (B, state_dim)
        - actions:     shape (B, action_dim)
        - rewards:     shape (B, 1)
        - next_states: shape (B, state_dim)
        - dones:       shape (B, 1)

        Note:
        - Return tensors directly on self.device so training code can use them
          immediately.
        """
        states = None
        actions = None
        rewards = None
        next_states = None
        dones = None

        ####################################################################
        # TODO:
        # Step 1) Randomly choose indices from valid stored data
        # Step 2) Gather the corresponding numpy arrays
        # Step 3) Convert them to torch.float32 tensors
        # Step 4) Move them to self.device
        ####################################################################
        assert self.size >= batch_size, "Not enough samples in buffer"

        idx = np.random.randint(0, self.size, size=batch_size)

        states = self.states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        next_states = self.next_states[idx]
        dones = self.dones[idx]

        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()
        rewards = torch.from_numpy(rewards).float()
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(dones).float()

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size


class SAC:
    """
    Soft Actor-Critic agent.

    Big picture:
    - SAC is a continuous-control reinforcement learning algorithm.
    - It learns:
        1) an actor: how to choose actions
        2) two critics: how good actions are
        3) optionally alpha: how much randomness / entropy to encourage

    Core idea:
    - Learn actions that have high long-term return
    - But also keep the policy sufficiently stochastic during training
    - So SAC optimizes both reward and entropy

    Why two critics?
    - Q-learning methods often overestimate Q values
    - SAC reduces this problem by learning twin critics and using:
          min(Q1, Q2)

    Why this class is generic:
    - It does not assume what your state means physically
    - The state can be joint-space, workspace, or mixed observations
    - It only assumes continuous vectors
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low,
        action_high,
        actor_hidden_dims: Tuple[int, ...] = (256, 256),
        critic_hidden_dims: Tuple[int, ...] = (256, 256),
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        init_alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize all networks, target networks, and optimizers.

        What happens here:
        - store environment action range
        - create actor
        - create two critics
        - create target critics
        - create optimizers
        - optionally create a learnable entropy coefficient alpha

        Why action scaling is needed:
        - The actor naturally outputs normalized actions in [-1, 1]
        - But the real environment may want a different range, such as:
              torque in [-2, 2]
              velocity in [-0.5, 0.5]
        - So SAC keeps a normalized action space internally, then rescales
          actions before sending them to the environment

        Why alpha exists:
        - Alpha controls the tradeoff between:
            reward maximization
            entropy maximization
        - Larger alpha -> more exploration / randomness
        - Smaller alpha -> more greedy behavior
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.auto_entropy_tuning = auto_entropy_tuning
        self.device = torch.device(device)

        ####################################################################
        # TODO:
        # Step 1) Store action_low and action_high as tensors on device
        #
        # Step 2) Precompute:
        #   action_scale = (high - low) / 2
        #   action_bias  = (high + low) / 2
        #
        # Why?
        # - The actor outputs normalized actions in [-1, 1]
        # - The environment may expect [low, high]
        # - These values let us map back and forth easily
        #
        # Step 3) Create:
        #   actor
        #   critic1, critic2
        #   target_critic1, target_critic2
        #
        # Step 4) Copy critic parameters into target critics
        #
        # Why target critics?
        # - Bellman targets become more stable when using slowly updated
        #   target networks
        #
        # Step 5) Create optimizers for actor and critics
        #
        # Step 6) Entropy coefficient alpha
        # - If auto_entropy_tuning=True:
        #     learn log_alpha automatically
        # - Else:
        #     use a fixed scalar alpha
        ####################################################################
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        self.actor = Actor(state_dim, action_dim, actor_hidden_dims).to(self.device)
        self.critic1 = Critic(state_dim, action_dim, critic_hidden_dims).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, critic_hidden_dims).to(self.device)
        self.target_critic1 = Critic(state_dim, action_dim, critic_hidden_dims).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim, critic_hidden_dims).to(self.device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        if self.auto_entropy_tuning:
            self.log_alpha = torch.tensor(
                np.log(init_alpha),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
            self.target_entropy = -float(action_dim) if target_entropy is None else target_entropy
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self.target_entropy = None
            self.alpha = torch.tensor(init_alpha, dtype=torch.float32, device=self.device)
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

    @property
    def current_alpha(self) -> torch.Tensor:
        """
        Return the current entropy coefficient alpha.

        What alpha means:
        - Alpha controls how much SAC values policy entropy.
        - In the actor loss:
              alpha * log_pi - Q
          alpha decides how strongly randomness is encouraged.

        Cases:
        - If auto entropy tuning is enabled:
            alpha = exp(log_alpha)
          We optimize log_alpha because it is numerically more stable
          and guarantees alpha stays positive.

        - If auto entropy tuning is disabled:
            alpha is just a fixed constant.
        """
        ####################################################################
        # TODO:
        # Return alpha as a tensor on self.device
        ####################################################################
        return self.log_alpha.exp() if self.auto_entropy_tuning else self.alpha
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

    def _scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized action to environment action range.

        What this does:
        - Input action is assumed to be in [-1, 1]
        - Output action is mapped into [action_low, action_high]

        Why this is needed:
        - The actor works in a normalized action space
        - The environment usually expects real physical units

        Example:
        - normalized action = 0.5
        - env torque range = [-2, 2]
        - scaled action = 1.0
        """
        scaled_action = None

        ####################################################################
        # TODO:
        # Implement:
        #   scaled_action = action * action_scale + action_bias
        ####################################################################
        scaled_action = action * self.action_scale + self.action_bias
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

        return scaled_action

    # def _unscale_action(self, action: torch.Tensor) -> torch.Tensor:
    #     """
    #     Convert environment action back to normalized [-1, 1] range.

    #     Why this may be needed:
    #     - Sometimes replay buffer stores actions in environment units
    #     - But the critics may be designed to work with normalized actions
    #     - Then we need to undo the scaling before feeding actions into critics
    #     """
    #     unscaled_action = None

    #     ####################################################################
    #     # TODO:
    #     # Implement:
    #     #   unscaled_action = (action - action_bias) / action_scale
    #     ####################################################################
    #     unscaled_action = (action - self.action_bias) / self.action_scale
    #     ####################################################################
    #     #                          END OF YOUR CODE                        #
    #     ####################################################################

    #     return unscaled_action

    def select_action(self, state, deterministic: bool = False) -> np.ndarray:
        """
        Choose one action for interacting with the environment.

        Training vs testing:
        - During training, we usually want stochastic actions for exploration
        - During evaluation, we often want deterministic actions

        Procedure:
        - Convert input state to a tensor
        - Add batch dimension if needed
        - Query actor.sample(...)
        - Use:
            sampled action      if deterministic=False
            mean_action         if deterministic=True
        - Scale action from [-1, 1] to environment range
        - Return numpy arrays of scaled and normalized actions

        Output:
        - actions in environment scale and normalized scale
        """
        action = None

        ####################################################################
        # TODO:
        # Step 1) Convert state to tensor on device
        # Step 2) Add batch dimension if state is 1D
        # Step 3) Use actor.sample()
        # Step 4) Pick stochastic or deterministic action
        # Step 5) Scale to environment range
        # Step 6) Remove batch dimension and convert to numpy (both scaled and normalized)
        ####################################################################
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        if state.ndim == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            sampled_action, _, mean_action = self.actor.sample(state)
            action_norm = mean_action if deterministic else sampled_action
            action_env = self._scale_action(action_norm)

        action_env = action_env.squeeze(0).cpu().numpy()
        action_norm = action_norm.squeeze(0).cpu().numpy()
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

        return action_env, action_norm

    def compute_critic_loss(
        self, batch
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Compute the losses for the two critics.

        What the critics are learning:
        - Each critic should predict the Bellman target:
              y = r + gamma * (1 - done) * target_value_of_next_state

        In SAC, that target value is:
            min(target_q1, target_q2) - alpha * next_log_prob

        Why subtract alpha * next_log_prob?
        - SAC does not only maximize reward
        - It also values entropy
        - So the target includes both action value and policy entropy

        Why use target critics?
        - To make the bootstrapped target more stable

        Why use min(target_q1, target_q2)?
        - To reduce overestimation bias
        """
        critic1_loss = None
        critic2_loss = None
        info = {}

        ####################################################################
        # TODO:
        # Step 1) Unpack the batch:
        #   states, actions, rewards, next_states, dones
        #
        # Step 2) Build the Bellman target without gradients:
        #   - sample next_action and next_log_prob from actor(next_states)
        #   - evaluate target critics on next_state and next_action
        #   - take min(target_q1, target_q2)
        #   - subtract alpha * next_log_prob
        #   - compute:
        #         bellman_target = rewards + gamma * (1 - dones) * target_q
        #
        # Step 3) Evaluate current critics on current states and actions
        #
        # Step 4) Compute MSE losses against the Bellman target
        #
        # Step 5) Save useful debug info
        ####################################################################
        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_action)
            target_q2 = self.target_critic2(next_states, next_action)
            target_q = torch.min(target_q1, target_q2) - self.current_alpha * next_log_prob
            bellman_target = rewards + self.gamma * (1 - dones) * target_q
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(q1, bellman_target)
        critic2_loss = F.mse_loss(q2, bellman_target)

        info["critic1_loss"] = critic1_loss.item()
        info["critic2_loss"] = critic2_loss.item()
        info["q1_mean"] = q1.mean().item()
        info["q2_mean"] = q2.mean().item()
        info["target_q_mean"] = bellman_target.mean().item()
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

        return critic1_loss, critic2_loss, info

    def compute_actor_loss(
        self, states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Compute the actor loss.

        What the actor is trying to do:
        - Choose actions that critics think are good
        - But also keep the policy from becoming too deterministic too early

        SAC actor objective:
            actor_loss = E[ alpha * log_pi - min(q1, q2) ]

        How to interpret this:
        - min(q1, q2) should be large
          -> good actions reduce the loss
        - alpha * log_pi adds entropy pressure
          -> prevents the policy from collapsing too early

        Intuition:
        - "Pick actions with high value"
        - "But keep enough randomness while learning"
        """
        actor_loss = None
        log_pi = None
        info = {}

        ####################################################################
        # TODO:
        # Step 1) Sample actions from the current policy at these states
        # Step 2) Compute q1 and q2 of those sampled actions
        # Step 3) Take q = min(q1, q2)
        # Step 4) Compute:
        #           actor_loss = mean(alpha * log_pi - q)
        #
        # Why use sampled actions here?
        # - Because we want to optimize the current stochastic policy itself
        ####################################################################
        actions, log_pi, _ = self.actor.sample(states)
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        q = torch.min(q1, q2)
        actor_loss = torch.mean(self.current_alpha * log_pi - q)

        info["actor_loss"] = actor_loss.item()
        info["log_pi_mean"] = log_pi.mean().item()
        info["q_mean"] = q.mean().item()
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

        return actor_loss, log_pi, info

    def compute_alpha_loss(self, log_pi: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for learning alpha automatically.

        What alpha tuning is trying to do:
        - If the policy is too deterministic, increase alpha
          so entropy is encouraged more
        - If the policy is too random, decrease alpha

        Standard SAC formula:
            alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()

        Why target_entropy is usually negative:
        - We want a certain amount of randomness in the action distribution
        - A common default is:
              target_entropy = -action_dim

        Why detach(log_pi + target_entropy)?
        - When updating alpha, we only want to change alpha
        - We do NOT want this loss to also update the actor
        """
        alpha_loss = None

        ####################################################################
        # TODO:
        # Implement the standard SAC alpha loss
        ####################################################################
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

        return alpha_loss

    def update(self, replay_buffer: ReplayBuffer, batch_size: int) -> Dict[str, float]:
        """
        Run one SAC training step.

        Full training flow:
        1) sample a batch from replay buffer
        2) update critic1 and critic2
        3) update actor
        4) update alpha if auto-entropy tuning is enabled
        5) softly update target critics

        Why this order is common:
        - Critics learn targets first
        - Actor then improves using the current critics
        - Targets are updated slowly at the end
        """
        info = {}

        ####################################################################
        # TODO:
        # Step 1) Sample one mini-batch from replay buffer
        #
        # Step 2) Critic update:
        #   - compute critic losses
        #   - zero_grad
        #   - backward
        #   - optimizer step
        #
        # Step 3) Actor update:
        #   - compute actor loss
        #   - zero_grad
        #   - backward
        #   - optimizer step
        #
        # Step 4) Alpha update if enabled
        #
        # Step 5) Softly update target critics
        #
        # Step 6) Return scalar diagnostics in info
        ####################################################################
        info = {}

        mini_batch = replay_buffer.sample(batch_size)

        # critic update
        critic1_loss, critic2_loss, critic_info = self.compute_critic_loss(mini_batch)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        info.update(critic_info)

        # actor update
        states = mini_batch[0]

        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        actor_loss, log_pi, actor_info = self.compute_actor_loss(states)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True

        info.update(actor_info)

        # alpha update
        if self.auto_entropy_tuning:
            alpha_loss = self.compute_alpha_loss(log_pi)

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            info["alpha_loss"] = alpha_loss.item()
            info["alpha"] = self.current_alpha.item()

        # soft target update
        self.soft_update_targets()
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

        return info

    def soft_update_targets(self) -> None:
        """
        Soft-update the target critics.

        Formula:
        - target_param = tau * param + (1 - tau) * target_param

        Why not copy parameters directly every step?
        - If targets change too fast, Bellman targets become unstable
        - Soft update makes training smoother
        """
        ####################################################################
        # TODO:
        # Soft-update target_critic1 from critic1
        # Soft-update target_critic2 from critic2
        ####################################################################
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

    def train_mode(self) -> None:
        """
        Put networks in training mode.

        Why this matters:
        - Some layers behave differently in train/eval mode
          (e.g. dropout, batchnorm)
        - Even if your current network does not use them,
          keeping train/eval mode handling is still good practice
        """
        ####################################################################
        # TODO:
        # Call .train() on relevant networks
        ####################################################################
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        self.target_critic1.train()
        self.target_critic2.train()
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

    def eval_mode(self) -> None:
        """
        Put networks in evaluation mode.

        Use this when:
        - you want deterministic evaluation
        - you are validating performance
        - you are saving rollouts or videos
        """
        ####################################################################
        # TODO:
        # Call .eval() on relevant networks
        ####################################################################
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.target_critic1.eval()
        self.target_critic2.eval()
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

    def save(self, path: str) -> None:
        """
        Save the training state.

        What should usually be saved:
        - model weights
        - target network weights
        - optimizer states
        - alpha-related states if used

        Why save optimizer states too?
        - So training can resume smoothly, not just model inference
        """
        ####################################################################
        # TODO:
        # Build a checkpoint dictionary and save it with torch.save
        ####################################################################
        checkpoint = {
            # model parameters
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),

            # optimizer status
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),

            # hyper parameters
            "gamma": self.gamma,
            "tau": self.tau,
        }

        # alpha
        if self.auto_entropy_tuning:
            checkpoint["log_alpha"] = self.log_alpha.detach().cpu()
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()
            checkpoint["target_entropy"] = self.target_entropy
        else:
            checkpoint["alpha"] = self.alpha.detach().cpu()

        torch.save(checkpoint, path)
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################

    def load(self, path: str) -> None:
        """
        Load a previously saved training state.

        What this restores:
        - network parameters
        - target network parameters
        - optimizer states
        - alpha states if present

        Why map_location=self.device?
        - So checkpoints can be loaded correctly whether they were saved
          on CPU or GPU
        """
        ####################################################################
        # TODO:
        # Load checkpoint and restore all states
        ####################################################################
        checkpoint = torch.load(path, map_location=self.device)

        # -------- model parameters --------
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2"])

        # -------- optimizer --------
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])

        def _move_optimizer_to_device(optimizer, device):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        _move_optimizer_to_device(self.actor_optimizer, self.device)
        _move_optimizer_to_device(self.critic1_optimizer, self.device)
        _move_optimizer_to_device(self.critic2_optimizer, self.device)

        # -------- alpha--------
        if self.auto_entropy_tuning and "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
            _move_optimizer_to_device(self.alpha_optimizer, self.device)
            self.target_entropy = checkpoint["target_entropy"]
        elif "alpha" in checkpoint:
            self.alpha = checkpoint["alpha"].to(self.device)

        print(f"Loaded checkpoint from {path}")
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################
