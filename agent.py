import torch
from config import State, AgentConfig
from q_model import QNetwork
import torch.nn as nn

torch.manual_seed(42)

class Agent:
    def __init__(self):
        self.device = AgentConfig.device
        self.n_actions = AgentConfig.n_actions
        self.gamma = AgentConfig.gamma
        self.state_dim = AgentConfig.state_dim

        self.q_net = QNetwork(self.state_dim, self.n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=AgentConfig.lr)
        self.loss_fn = nn.MSELoss()


    def select_action(self, state: torch.Tensor) -> int:
        """
        state: tensor shape (state_dim,)
        returns: action index (0, 1, or 2)
        """
        state = state.to(self.device).unsqueeze(0)  # -> shape (1, state_dim)

        # ε-soft / ε-greedy
        if torch.rand(1).item() < self.epsilon:
            # Explore: random action
            action = torch.randint(0,self.n_actions, (1,)).item()
        else:
            # Exploit: argmax_a Q(s,a)
            with torch.no_grad():
                q_values = self.q_net(state)         # shape (1, n_actions)
                action = int(q_values.argmax(dim=1)) # scalar int

        return action


    def learn_batch(
            self,
            states,
            actions,
            rewards,
            next_states,
            dones):
        """
        Batch Q-learning update.
        shapes:
            states:      (B, state_dim)
            actions:     (B,)
            rewards:     (B,)
            next_states: (B, state_dim)
            dones:       (B,)
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 1. Compute Q(s,a) for each sample
        q_values = self.q_net(states)  # (B, n_actions)

        # Select Q(s_i, a_i)
        q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # shape (B,)

        # 2. Compute targets
        with torch.no_grad():
            next_q = self.q_net(next_states).max(dim=1).values  # (B,)
            targets = rewards + self.gamma * next_q * (1 - dones)

        # 3. Loss (MSE over batch)
        loss = self.loss_fn(q_selected, targets)

        # 4. Backprop + update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()







if __name__ == "__main__":
    print("=== SANITY CHECK START ===")

    # 1. Create agent
    agent = Agent()
    print("Agent created.")
    print("Q-network:", agent.q_net)

    # 2. Construct random state (tensor)
    state_dim = agent.state_dim
    dummy_state = torch.randn(state_dim)
    print("Dummy state:", dummy_state)

    # 3. Check forward pass
    dummy_state_batch = dummy_state.unsqueeze(0)  # shape (1, state_dim)
    q_vals = agent.q_net(dummy_state_batch)
    print("Q-values shape:", q_vals.shape)
    print("Q-values:", q_vals)

    assert q_vals.shape == (1, agent.n_actions), "Q-network output shape is wrong!"

    # 4. Test select_action()
    agent.epsilon = 0.0   # force greedy
    action = agent.select_action(dummy_state)
    print("Selected action:", action)


    assert isinstance(action, int), "Action must be an int!"
    assert 0 <= action < agent.n_actions, "Action out of range!"

    # 5. Dummy training step
    print("\nRunning 1 dummy training step...")

    # Make dummy target Q
    target_q = torch.randn_like(q_vals)  # random target

    loss_fn = agent.loss_fn
    loss = loss_fn(q_vals, target_q)

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    print("Loss:", loss.item())

    # 6. Check parameters updated
    with torch.no_grad():
        new_q_vals = agent.q_net(dummy_state_batch)
    print("Q-values after 1 update:", new_q_vals)

    print("=== SANITY CHECK PASSED ===")



