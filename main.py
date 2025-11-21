import torch
from environmnet import Environment   # your class file
from config import EnvConfig

# Create environment
env = Environment()

# Reset manually or do your own reset method
print("Initial state:", env.reset())

# Try each action: 0 = short, 1 = flat, 2 = long
actions = [0, 1, 2, 2, 0]   # try various transitions

for t, a in enumerate(actions):
    print(f"\n--- Step {t+1} | action = {a} ---")
    next_state, reward = env.step(a)

    print("Next state:", next_state)
    print("Reward    :", reward.item())
    print("Price     :", env.S.item())
    print("Position  :", env.pos.item())
