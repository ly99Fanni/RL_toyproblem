import torch
from config import EnvConfig, State, Action, Reward


class Environment:
    def __init__(self):
        self.vol = torch.tensor(EnvConfig.vol, dtype=torch.float32)
        self.kappa = torch.tensor(EnvConfig.kappa, dtype=torch.float32)
        self.price_mean = torch.tensor(EnvConfig.price_mean, dtype=torch.float32)
        self.half_ba = torch.tensor(EnvConfig.half_ba, dtype=torch.float32)
        self.risk_av = torch.tensor(EnvConfig.risk_av, dtype=torch.float32)

        # internal state
        self.S = None   # current price S_t
        self.pos= None # current position pos_t
        self.state= None        # current discrete state: 0=DOWN, 1=FLAT, 2=UP
        self.price_list: list[float] = []


    # -------- Price dynamics --------
    def generate_next_price(self, prev_price: torch.Tensor | float) -> torch.Tensor:
        # ensure tensor type
        if not isinstance(prev_price, torch.Tensor):
            prev_price = torch.tensor(prev_price, dtype=torch.float32)

        eps = torch.randn_like(prev_price)
        next_price = prev_price + self.vol * eps - self.kappa * (prev_price - self.price_mean)
        return next_price

    def generate_price_series(self, T: int) -> torch.Tensor:
        S = torch.zeros(T, dtype=torch.float32)
        S[0] = EnvConfig.S0

        for i in range(1, T):
            S[i] = self.generate_next_price(S[i - 1])

        return S


    def step(self, action: Action) -> tuple[State, Reward]:
        """
        One trading day:

        1. Choose new position pos_t from action.
        2. Pay friction for changing position: half_ba * |pos_t - pos_{t-1}|
        3. Simulate new price S_t
        4. Receive reward R_t as defined.

        Returns:
        -------
        next_state : discrete trend state in {0=DOWN, 1=FLAT, 2=UP}
        reward     : scalar tensor
        """

        # 0) Current price and position
        prev_price = self.S
        prev_pos = self.pos

        # 1) Map action to position
        new_pos = EnvConfig.map_action_to_position_tensor(action)

        # 2) Transaction cost
        d_pos = torch.abs(new_pos - prev_pos)
        friction = self.half_ba * d_pos

        # 3) Volatility penalty
        penalty = self.risk_av * (self.vol * new_pos) ** 2

        # 4) Simulate new price and compute price change
        new_price = self.generate_next_price(prev_price)
        d_price = new_price - prev_price

        # 5) Calculate the reward
        reward = new_pos * d_price - friction - penalty

        # 6) Update internal state
        trend_state = self._price_to_state(prev_price, new_price)

        self.S = new_price
        self.pos = new_pos
        self.state = trend_state

        # Return discrete state + reward
        return self.state, reward

    def reset(self) -> tuple[State, torch.Tensor, torch.Tensor]:
        """
        Reset the environment to the initial state.

        Returns:
        --------
        state : initial trend state (we treat it as FLAT = 1)
        S     : initial price tensor
        pos   : initial position tensor
        """
        self.S = torch.tensor(EnvConfig.S0, dtype=torch.float32)
        self.pos = EnvConfig.map_action_to_position_tensor(EnvConfig.a0)
        self.state = 1

        return self.state, self.S, self.pos

    # -------- State helpers --------
    def _get_state_vector(self) -> torch.Tensor:
        """
        Optional: continuous observation vector [price, position].
        You can use this later if you move to function approximation.
        """
        return torch.tensor([self.S.item(), self.pos.item()], dtype=torch.float32)

    def _price_to_state(self, prev_price: torch.Tensor, new_price: torch.Tensor) -> int:
        """
        Map price change into 3 discrete states:
        0 = DOWN, 1 = FLAT, 2 = UP
        """
        ret = torch.log(new_price / prev_price)

        # threshold scaled by vol; tune 0.25 factor as you like
        eps = 0.01 * self.vol

        ret_val = ret.item()
        eps_val = eps.item()

        if ret_val > eps_val:
            return 2  # UP
        elif ret_val < -eps_val:
            return 0  # DOWN
        else:
            return 1  # FLAT


if __name__ == "__main__":
    torch.manual_seed(42)

    # ============ SANITY CHECK ============
    env = Environment()
    init_state, init_S, init_pos = env.reset()
    print("Initial state (trend, S, pos):", init_state, init_S.item(), init_pos.item())

    # actions should be integer-coded, not float
    actions = torch.tensor([0, 1, 2, 2, 0], dtype=torch.int64)

    for t, a in enumerate(actions):
        print(f"\n--- Step {t + 1} | action = {a.item()} ---")
        next_state, reward = env.step(a.item())

        print("Next trend state:", next_state)       # 0/1/2
        print("Reward          :", reward.item())
        print("Price           :", env.S.item())
        print("Position        :", env.pos.item())
