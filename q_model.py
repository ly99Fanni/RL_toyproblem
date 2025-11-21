import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: shape (state_dim,) or (batch_size, state_dim)
        returns: Q-values, shape (n_actions,) or (batch_size, n_actions)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.out(x)
        return q_values









#####################################################################################

def test_forward_pass():
    print("=== Testing QNetwork forward pass ===")

    state_dim = 2
    n_actions = 3

    net = QNetwork(state_dim, n_actions)

    # Fake input state
    state = torch.tensor([100.0, 1.0])  # shape (2,)

    # Add batch dimension
    state_batched = state.unsqueeze(0)  # shape (1, 2)

    # Forward pass
    q_values = net(state_batched)

    print("Input shape:      ", state_batched.shape)
    print("Output shape:     ", q_values.shape)
    print("Q-values:         ", q_values)

    assert q_values.shape == (1, n_actions), "Output shape is wrong"
    print("✔ Forward pass test passed.\n")


def test_backward_pass():
    print("=== Testing backward pass ===")

    state_dim = 2
    n_actions = 3

    net = QNetwork(state_dim, n_actions)

    # Fake batch of input states
    state = torch.tensor([[100.0, 1.0]], requires_grad=True)

    # Forward pass
    q_values = net(state)   # shape (1,3)

    # Construct a fake target
    target = torch.tensor([[1.0, 0.0, -1.0]])

    # Compute MSE loss
    loss = ((q_values - target)**2).mean()

    print("Loss value:", loss.item())

    # Backprop
    loss.backward()

    # Check if gradients exist
    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        print(f"✔ Gradient OK for {name}, shape {param.grad.shape}")

    print("✔ Backward pass test passed.\n")


if __name__ == "__main__":
    test_forward_pass()
    test_backward_pass()

