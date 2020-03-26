import time

from algorithm.multi.ddpg import ddpgAgent
from make_env import make_env
import matplotlib.pyplot as plt

from config import config, params

env = make_env(config["scenario"])

action_dim = env.action_space[0].n
state_dim = env.observation_space[0].shape[0]

agent = ddpgAgent(state_dim, action_dim, params)
agent.load_model(config)
agent.actor.eval()

state = env.reset()
reward_n = []
step_num = 0
while step_num < config["test_step"]:
    step_num = step_num + 1
    env.render()
    time.sleep(0.01)
    action = agent.act(state)
    state, reward, is_done, _ = env.step(action)
    reward_n.append(reward)

plt.plot(reward_n)
plt.show()
