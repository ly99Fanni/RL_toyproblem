import torch
from config import AgentConfig
from environmnet import Environment


class Agent2:
    def __init__(self):
        self.epsilon = AgentConfig.epsilon
        self.device = AgentConfig.device
        self.n_actions = AgentConfig.n_actions
        self.gamma = AgentConfig.gamma
        self.n_states = AgentConfig.n_states
        self.lr = AgentConfig.lr
        self.batch_size = AgentConfig.batch_size

        # Q-table lives on the agent, persists across episodes
        self.q_values = torch.zeros((self.n_states, self.n_actions), dtype=torch.float32, device=self.device)

        # simple replay buffer (lists of scalars)
        self._states: list[int] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._next_states: list[int] = []

    def get_action(self, state: int) -> int:
        """
        Epsilon-greedy policy for a discrete state.
        state: 0:DOWN, 1: FLAT, 2:UP
        """
        if torch.rand(1).item() < self.epsilon:
            # Explore
            action = torch.randint(0, self.n_actions, (1,)).item()
        else:
            # Exploit
            action = torch.argmax(self.q_values[state]).item()
        return action

    # --------- replay buffer helpers ---------
    def _store_transition(self, s: int, a: int, r: float, s_next: int) -> None:
        self._states.append(s)
        self._actions.append(a)
        self._rewards.append(r)
        self._next_states.append(s_next)

    def _can_sample(self) -> bool:
        return len(self._states) >= self.batch_size

    def _sample_minibatch(self, batch_size):
        """
        Returns:
        states_b      : (B,) long
        actions_b     : (B,) long
        rewards_b     : (B,) float
        next_states_b : (B,) long
        """
        N = len(self._states)
        B = batch_size

        idx = torch.randint(0, N, (B,))

        states_b = torch.tensor(
            [self._states[i] for i in idx],
            dtype=torch.long,
            device=self.device,
        )
        actions_b = torch.tensor(
            [self._actions[i] for i in idx],
            dtype=torch.long,
            device=self.device,
        )
        rewards_b = torch.tensor(
            [self._rewards[i] for i in idx],
            dtype=torch.float32,
            device=self.device,
        )
        next_states_b = torch.tensor(
            [self._next_states[i] for i in idx],
            dtype=torch.long,
            device=self.device,
        )

        return states_b, actions_b, rewards_b, next_states_b

    # --------- training loop ---------
    def train(self,env: Environment,num_episodes: int, max_steps_per_episode: int, batch_size: int) -> list[float]:
        """
        Tabular Q-learning with replay minibatches.

        - Outer loop over episodes
        - Inner loop: interact with env, store transitions
        - Once replay buffer has at least batch_size samples, do a
          vectorized minibatch update.

        Returns list of total reward per episode.
        """
        total_rewards_per_episode: list[float] = []
        prices_per_episode: list[float] = []

        for ep in range(num_episodes):
            # Reset environment at start of each episode
            state, S0, _ = env.reset()
            total_reward = 0.0

            episode_price_series = [S0.clone()]

            for t in range(max_steps_per_episode):
                # 1) Choose action
                action = self.get_action(int(state))

                # 2) Take action and observe outcome
                next_state, reward = env.step(action)
                episode_price_series.append(env.S.clone())

                # ensure reward is a tensor on the same device as Q
                if not isinstance(reward, torch.Tensor):
                    reward = torch.tensor(reward, dtype=torch.float32)
                reward = reward.to(self.device)

                # accumulate reward as float for logging
                total_reward += float(reward.item())

                # 3) Store transition in replay buffer
                self._store_transition(
                    s=int(state),
                    a=int(action),
                    r=float(reward.item()),
                    s_next=int(next_state),
                )

                # 4) If enough samples, do ONE minibatch update
                if self._can_sample():
                    states_b, actions_b, rewards_b, next_states_b = self._sample_minibatch(batch_size)

                    # Q(next_state, ·) -> max over actions, shape (B,)
                    best_next_q = self.q_values[next_states_b].max(dim=1).values

                    # TD target: r + gamma * max_a' Q(s', a')
                    td_target = rewards_b + self.gamma * best_next_q  # (B,)

                    # Current Q(s,a) for the batch
                    q_sa = self.q_values[states_b, actions_b]  # (B,)

                    # TD error and update
                    td_error = td_target - q_sa  # (B,)
                    self.q_values[states_b, actions_b] = q_sa + self.lr * td_error

                # 5) Move to next state
                state = next_state

            total_rewards_per_episode.append(total_reward)
            prices_per_episode.append(torch.stack(episode_price_series))

        return total_rewards_per_episode, prices_per_episode


if __name__ == "__main__":
    torch.manual_seed(42)

    env = Environment()
    agent = Agent2()

    print(f"--- Running Q-Learning ---")
    print(
        f"Episodes: {AgentConfig.num_episodes}, "
        f"Steps/episode: {AgentConfig.max_steps_per_episode}, "
        f"Minibatch size: {AgentConfig.batch_size}, "
        f"Alpha: {AgentConfig.lr}, Gamma: {AgentConfig.gamma}, "
        f"Epsilon: {AgentConfig.epsilon}"
    )

    rewards, price_series = agent.train(
        env,
        num_episodes=AgentConfig.num_episodes,
        max_steps_per_episode=AgentConfig.max_steps_per_episode,
        batch_size= AgentConfig.batch_size
    )

    for ep in range(AgentConfig.num_episodes):
        r = rewards[ep]
        series = [round(x, 2) for x in price_series[ep].tolist()]

        print(f"Episode {ep + 1} - Total Reward: {r:.4f}")
        print("Price series:", series)
        print("-" * 40)

