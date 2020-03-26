import time

import gym

from algorithm.single.policy_gradient import vanilla_policy_gradient
from config import *
import numpy as np

# Pendulum-v0 CartPole-v1
env_name = 'CartPole-v1'
env = gym.make(env_name)

params = {
    "value_forms": "discount_reward",
    "hidden_dim": 64,
    "batch_size": 4096,
    "steps_max": 100,
    "log_freq": 100,
    "capacity": 1e6,
    "episode_max": 10000,
    "is_action_discrete": isinstance(env.action_space, gym.spaces.Discrete)
}

if params["is_action_discrete"]:
    action_dim = env.action_space.n
else:
    action_dim = env.action_space.shape[0]

state_dim = env.observation_space.shape[0]

agent = vanilla_policy_gradient(state_dim, action_dim, params)
value_function = agent.value_funtion()

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

    agent.put(state, action, value_function(reward))
    agent.update(params["batch_size"])

    log_time += time.time() - old_time
    return_sum += np.array(reward).sum()

    if episode % params["log_freq"] == 0:
        print("episode: %i----time: %.3f; mean_return %.3f" % (episode, log_time, return_sum/params["log_freq"]))
        log_time = 0
        return_sum = 0
