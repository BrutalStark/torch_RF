import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

import numpy as np


def naive_value(rewards):
    return (np.ones((len(rewards),)) * np.array(rewards).sum()).tolist()


def reward_to_go(rewards):
    prev = 0
    reward_after_action = []
    for reward in reversed(rewards):
        reward_after_action.append(reward + prev)
        prev = reward_after_action[-1]
    return reversed(reward_after_action)


def discount_reward(rewards, gamma=0.99):
    r = np.array([gamma ** i * rewards[i] for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    return (r - r.mean()).tolist()


class policy_net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(policy_net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        out = self.relu(self.fc1(state))
        out = self.relu(self.fc2(out))
        out = self.tanh(self.fc3(out))
        return out


class vanilla_policy_gradient:
    def __init__(self, state_dim, action_dim, params):
        self.policy = policy_net(state_dim, action_dim, params["hidden_dim"])
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=0.01)

        self.batch_size = params["batch_size"]
        self.is_action_discrete = params["is_action_discrete"]

        self.state = []
        self.action = []
        self.R = []

        self.value_forms = params["value_forms"]

    def value_funtion(self):
        if self.value_forms is "naive":
            return naive_value
        if self.value_forms is "reward_to_go":
            return reward_to_go
        if self.value_forms is "discount_reward":
            return discount_reward

    def clear(self):
        self.state = []
        self.action = []
        self.R = []

    def put(self, state, action, R):
        self.state += state
        self.action += action
        self.R += R

    def get_discrete_policy(self, state):
        logits = self.policy(state)
        return Categorical(logits=logits)

    def get_continous_policy(self, state):
        means, std = self.policy(state)
        return Normal(loc=means, scale=std)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float)
        if self.is_action_discrete:
            dist = self.get_discrete_policy(state)
        else:
            dist = self.get_continous_policy(state)
        return dist.sample().clamp(-1, 1).item()

    def update(self, batch_size):
        if len(self.action) < batch_size:
            return

        s0 = torch.tensor(self.state, dtype=torch.float)
        a0 = torch.tensor(self.action, dtype=torch.float)
        r1 = torch.tensor(self.R, dtype=torch.float)

        weight = r1

        self.policy_optimizer.zero_grad()
        if self.is_action_discrete:
            loss = -torch.mean(self.get_discrete_policy(s0).log_prob(a0) * weight)
        else:
            loss = -torch.mean(self.get_continous_policy(s0).log_prob(a0).sum(axis=-1) * weight)
        loss.backward()
        self.policy_optimizer.step()

        self.clear()
