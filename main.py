import torch
from environmnet import Environment
from agent2 import Agent2
from config import EnvConfig

torch.manual_seed(42)

market = Environment()
trader = Agent2()

batch_size = 10
episode = 5
steps = 100

rewards, price_series = trader.train(
    market,
    num_episodes= episode,
    max_steps_per_episode= steps,
    batch_size=batch_size
)


for ep in range(episode):
    r = rewards[ep]
    series = [round(x, 2) for x in price_series[ep].tolist()]

    print(f"Episode {ep + 1} - Total Reward: {r:.4f}")
    print("Price series:", series)
    print("-" * 40)
