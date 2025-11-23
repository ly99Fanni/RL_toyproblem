import torch
from environmnet import Environment
from agent import Agent
from config import EnvConfig


market = Environment()
trader = Agent()

batch_size = 10

#0 Set the initial state
state = market.reset()

for i in range(10):
    #1 choose action
    trader_action = trader.select_action(state)

    #2 set environment
    next_state, reward = market.step(trader_action)

    print('State: ', state)
    print('Trader action: ', trader_action)
    print('Reward: ', reward)
    print('Next state: ', next_state)
    print('Q values: ', trader.q_net(state), '\n')

    # 3 Update the state
    state = next_state

print('-'*40, '\n')
print('Cumulative reward: ', market.cum_reward)



