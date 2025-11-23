import torch
from environmnet import Environment
from agent import Agent
from config import EnvConfig


market = Environment()
trader = Agent()

state = market.reset()
trader_action = trader.select_action(state)
next_step = market.step(trader_action)
trader_learning = trader.learn_batch(states=state, actions=trader_action, rewards=next_step[1], next_states=next_step[0])

print('State: ', state)
print('Trader action: ', trader_action)
print('Next state: ', next_step)
print("One training step loss:", trader_learning)
print('Q value: ', trader.q_net(state))



