import time

import gym

from algorithm.single.policy_gradient import reward_to_go
from algorithm.single.ppo_clip import ppoClipAgent
from config import *
import numpy as np

# Pendulum-v0 CartPole-v1
env_name = 'Pendulum-v0'
env = gym.make(env_name)

params = {
    "hidden_dim": 64,
    "batch_size": 4096,
    "steps_max": 1000,
    "log_freq": 100,
    "capacity": 1e6,
    "episode_max": 1000,
    "K_epochs": 80,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # "device": "cpu"
}

action_dim = env.action_space.shape[0]

state_dim = env.observation_space.shape[0]

agent = ppoClipAgent(state_dim, action_dim, params)

step_sum = 0
log_time = 0
return_sum = 0
for episode in range(1, params["episode_max"]):
    state = []
    action = []
    reward = []

    s0 = env.reset()
    state.append(s0)

    episode_reward = 0
    step_episode = 0

    old_time = time.time()
    while True:
        step_sum += 1
        step_episode += 1

        a0 = agent.act(s0)
        s1, r1, done, _ = env.step(a0)
        action.append(a0)
        reward.append(r1)

        s0 = s1

        episode_reward += r1

        if done or step_episode > params["steps_max"]:
            break

        state.append(s0)

    if episode % params["log_freq"] == 0:
        print(episode, np.array(reward).mean())
    agent.put(state, action, reward_to_go(reward))
    agent.update()
