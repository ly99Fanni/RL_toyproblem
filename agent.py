import torch
from config import State, Action, AgentConfig
from q_model import QNetwork
import torch.nn as nn

torch.manual_seed(42)

class Agent:
    def __init__(self):
        self.epsilon = AgentConfig.epsilon
        self.device = AgentConfig.device
        self.n_actions = AgentConfig.n_actions
        self.gamma = AgentConfig.gamma
        self.state_dim = AgentConfig.state_dim

        self.q_net = QNetwork(self.state_dim, self.n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=AgentConfig.lr)
        self.loss_fn = nn.MSELoss()


    def select_action(self, state: torch.Tensor) -> Action:
        """
        state: tensor shape (state_dim,)
        returns: action index (0, 1, or 2)
        """
        state = state.to(self.device).unsqueeze(0)  # -> shape (1, state_dim)

        # ε-soft / ε-greedy
        if torch.rand(1).item() < self.epsilon:
            # Explore: random action
            action = torch.randint(0,self.n_actions, (1,))
        else:
            # Exploit: argmax_a Q(s,a)
            with torch.no_grad():
                q_values = self.q_net(state)         # shape (1, n_actions)
                action = q_values.argmax(dim=1)

        return action


    def learn_batch(
            self,
            states,
            actions,
            rewards,
            next_states):
        """
        Batch Q-learning update.
        shapes:
            states:      (B, state_dim)
            actions:     (B,)
            rewards:     (B,)
            next_states: (B, state_dim)
        """
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)

        if states.size(0) != 1 or next_states.size(0) != 1:
            states = states.to(self.device).unsqueeze(0)
            next_states = next_states.to(self.device).unsqueeze(0)

        # 1. Compute Q(s,a) for each sample
        q_values = self.q_net(states)  # (B, n_actions)

        # Select Q(s_i, a_i)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # shape (B,)

        # 2. Compute targets
        with torch.no_grad():
            next_q = self.q_net(next_states).max(dim=1).values  # (B,)
            targets = rewards + self.gamma * next_q

        # 3. Loss (MSE over batch)
        loss = self.loss_fn(q_selected, targets)

        # 4. Backprop + update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


if __name__ == "__main__":
    B = 1
    agent = Agent()

    # Fake batch
    states      = torch.randn(B, agent.state_dim)
    next_states = torch.randn(B,agent.state_dim)
    actions     = torch.randint(0, agent.n_actions, (B,))
    rewards     = torch.randn(B)
    dones       = torch.randint(0, 2, (B,)).float()  # 0 or 1

    print('States: {}'.format(states))
    print('Next state: {}'.format(next_states))
    print('Actions: {}'.format(actions))
    print('Rewards: {}'.format(rewards))
    print('dones: {}\n'.format(dones))

    # Test select_action
    s0 = states[0]
    a0 = agent.select_action(s0)
    print("Selected action:", a0)

    # Test learn_batch
    loss = agent.learn_batch(states, actions, rewards, next_states)
    print("One training step loss:", loss)







