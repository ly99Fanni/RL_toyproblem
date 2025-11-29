import torch

State = torch.Tensor
Action = torch.Tensor
Reward = torch.Tensor


class EnvConfig:
    S0:float = 100
    a0: float= 1
    price_mean: float = 100
    kappa: float = 0.1
    vol: float = 2
    half_ba: float = 1.0
    risk_av: float = 0.02
    gamma: float = 0.01

    @staticmethod
    def map_action_to_position_tensor(action: Action) -> torch.Tensor:
        """
        action: int tensor or scalar (0,1,2)
        returns: position tensor (-1.0, 0.0, 1.0) with same device/dtype
        """
        # ensure tensor
        if not isinstance(action, Action):
            action = torch.tensor(action, dtype=torch.long)

        # pos = action - 1  ->  0→-1, 1→0, 2→1
        pos = action.to(torch.float32) - 1.0
        return pos

class AgentConfig:
    epsilon: float = 0.25
    gamma: float = 0.99 # discount factor for one period
    lr: float = 0.0001
    device: str = 'cpu'
    n_actions: int = 3 # {0: short, 1: out, 2: long}
    n_states: int = 3 # down: 0, flat: 1, up: 2
    num_episodes: int= 5
    batch_size: int = 10
    max_steps_per_episode: int = 100


