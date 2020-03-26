import time

from make_env import make_env

from config import config, params
from algorithm.utils import get_multi_agent
import numpy as np

import matplotlib.pyplot as plt

algorithm = "ddpg"

env = make_env(config["scenario"])
env.reset()

state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n

agent = get_multi_agent(algorithm)(state_dim, action_dim, params)

step_sum = 0
rewards = []
episode_time = []
for episode in range(1, config["episodes"]):
    s0 = env.reset()
    old_time = time.time()
    episode_reward = 0
    for step in range(1, config["steps_max"]):
        # env.render()
        step_sum += 1

        a0 = agent.act(s0)
        s1, r1, done, _ = env.step(a0)

        old = time.time()
        agent.put(s0, a0, r1, s1)

        episode_reward += r1[0]
        s0 = s1

        agent.update(step_sum)
    rewards.append(episode_reward)
    episode_time.append(time.time() - old_time)

    # print training message
    if episode % config["log_episodes"] == 0:
        print("episode %i----mean_reward: %.3f; time: %.5f"
              % (episode, float(np.mean(rewards[episode - config["log_episodes"]: episode])),
                 float(np.sum(episode_time[episode - config["log_episodes"]: episode])))
              )

    # save model
    if episode % config["save_episodes"] == 0:
        print("saving")
        agent.save_model(config)

plt.plot(rewards)
# plt.plot(episode_time)
plt.xlabel("train epoch")
plt.ylabel("reward")
plt.title("train process on one good agent")
plt.show()

