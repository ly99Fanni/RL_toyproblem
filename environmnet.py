import torch
from config import EnvConfig, State, Action, Reward

torch.manual_seed(42)

class Environment:
    def __init__(self):
        self.vol = torch.tensor(EnvConfig.vol, dtype=torch.float32)
        self.kappa = torch.tensor(EnvConfig.kappa, dtype=torch.float32)
        self.price_mean = torch.tensor(EnvConfig.price_mean, dtype=torch.float32)
        self.half_ba = torch.tensor(EnvConfig.half_ba, dtype=torch.float32)
        self.risk_av = torch.tensor(EnvConfig.risk_av, dtype=torch.float32)
        self.vol = torch.tensor(EnvConfig.vol, dtype=torch.float32)

        # internal state
        self.S = None  # current price S_t
        self.pos = None  # current position pos_t
        self.t = None  # step counter
        self.cum_reward = None # cumulative reward

    def generate_next_price(self, prev_price) -> torch.Tensor:
        # ensure tensor type
        if not isinstance(prev_price, torch.Tensor):
            prev_price = torch.tensor(prev_price, dtype=torch.float32)

        eps = torch.randn_like(prev_price)
        next_price = prev_price + self.vol * eps - self.kappa*(prev_price-self.price_mean)
        return next_price

    def generate_price_series(self, T:int) -> torch.Tensor:
        S = torch.zeros(T)
        S[0] = EnvConfig.S0

        for i in range(1, T):
            S[i] = self.generate_next_price(S[i - 1])

        return S

    def step(self, action:Action) -> torch.Tensor:
        """
        One trading day:

        1. Choose new position pos_t from action.
        2. Pay friction for changing position: half_ba * |pos_t - pos_{t-1}|
        3. Simulate new price S_t
        4. Receive reward R_t as defined.


        Returns:
        next_state: shape (B, state_dim)
        reward:     shape (B,)

        """

        #0 Current price and position
        prev_price = self.S
        prev_pos = self.pos

        #1 Map action to position
        new_pos = EnvConfig.map_action_to_position_tensor(action)

        #2 Transaction cost
        d_pos = abs(new_pos- prev_pos)
        friction = self.half_ba * d_pos

        #3 Volatility penality
        penalty = self.risk_av * (self.vol * new_pos)**2

        #4 Simulate new price
        new_price = self.generate_next_price(prev_price)
        d_price = new_price - prev_price

        #5 Calculate the reward
        reward = new_pos * d_price - friction - penalty

        #6 Update internal state
        self.S = new_price
        self.pos = new_pos
        self.cum_reward += reward
        self.t += 1

        next_state = self._get_state()

        return next_state, reward

    def reset(self) -> torch.Tensor:
        self.S = EnvConfig.S0
        self.pos = EnvConfig.map_action_to_position_tensor(EnvConfig.a0)
        self.cum_reward = 0
        self.t = 0

        return self._get_state()

    def _get_state(self) -> torch.Tensor:
        # dev = (self.S - self.price_mean) / self.vol
        return torch.tensor([self.S, self.pos], dtype=torch.float32)



if __name__ == "__main__":
    # ============ SANITY CHECK ============
    env = Environment()
    state = env.reset()
    print("Initial state:", state)

    actions = torch.tensor([0, 1, 2, 2, 0], dtype=torch.float32)

    for t, a in enumerate(actions):
        print(f"\n--- Step {t + 1} | action = {a} ---")
        next_state, reward = env.step(a)

        print("Next state:", next_state)
        print("Reward    :", reward.item())
        print("Price     :", env.S.item())
        print("Position  :", env.pos.item())


