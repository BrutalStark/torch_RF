import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.optim as optim


def reward_to_go(rewards):
    prev = 0
    reward_after_action = []
    for reward in reversed(rewards):
        reward_after_action.append(reward + prev)
        prev = reward_after_action[-1]
    return reversed(reward_after_action)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        out = self.relu(self.fc1(state))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class ppoClipAgent:
    def __init__(self, state_dim, action_dim, params):
        self.device = params["device"]
        self.cov_mat = torch.diag(torch.full((action_dim,), 0.5 * 0.5))

        self.policy = Actor(state_dim, action_dim, params["hidden_dim"]).to(self.device)
        self.policy_old = Actor(state_dim, action_dim, params["hidden_dim"]).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.critic = Critic(state_dim, action_dim, params["hidden_dim"]).to(self.device)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=0.001)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=0.001)

        self.state = []
        self.action = []
        self.R = []
        self.old_log_prob = []

        self.K_epochs = params["K_epochs"]
        self.eps_clip = 0.2

    def clear(self):
        self.state = []
        self.action = []
        self.R = []
        self.old_log_prob = []

    def put(self, state, action, R):
        self.state += state
        self.action += action
        self.R += R

    def get_continuous_policy(self, state, net):
        if net is "old":
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action_mean = self.policy_old(state)
        else:
            action_mean = self.policy(state).to(self.device)
        dist = MultivariateNormal(action_mean, self.cov_mat)
        return dist

    def act(self, state):
        dist = self.get_continuous_policy(state, "old")
        action = dist.sample()
        self.old_log_prob.append(dist.log_prob(action).detach().tolist())
        return action.detach().to("cpu").numpy().clip(-1, 1)

    def update(self):
        # print((self.state[0]))
        # print((self.action[0]))
        # print((self.R[0]))
        # print((self.log_prob[0]))

        actions = torch.tensor(self.action, dtype=torch.float).to(self.device)

        rewards = torch.tensor(self.R, dtype=torch.float).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        state = torch.tensor(self.state, dtype=torch.float).to(self.device)
        value = self.critic(state).detach()

        advantage = rewards - value

        old_log_prob = torch.tensor(self.old_log_prob, dtype=torch.float).to(self.device)
        for _ in range(self.K_epochs):
            log_prob = self.get_continuous_policy(state, "new").log_prob(actions)
            ratio = torch.exp(log_prob - old_log_prob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            # update policy network
            loss = -torch.mean(torch.min(surr1, surr2))
            self.policy_optim.zero_grad()
            loss.backward()
            self.policy_optim.step()

            # update value network
            value = self.critic(state)
            loss = nn.MSELoss()(value, torch.tensor(rewards, dtype=torch.float).to(self.device))
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.clear()
