import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path


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
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        out = torch.cat([state, action], -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class ddpgAgent:
    def __init__(self, state_dim, action_dim, params):
        self.device = params["device"]

        self.actor = Actor(state_dim, action_dim, params["hidden_dim"]).to(self.device)

        if params["mode"] is "train":
            self.actor_target = Actor(state_dim, action_dim, params["hidden_dim"]).to(self.device)
            self.critic = Critic(state_dim, action_dim, params["hidden_dim"]).to(self.device)
            self.critic_target = Critic(state_dim, action_dim, params["hidden_dim"]).to(self.device)

            self.actor_optim = optim.Adam(self.actor.parameters(), lr=params["lr_actor"])
            self.critic_optim = optim.Adam(self.critic.parameters(), lr=params["lr_critic"])

            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

            self.update_freq = params["update_freq"]
            self.batch_size = params["batch_size"]
            self.gamma = params["gamma"]
            self.tau = params["tau"]

            self.buffer = []
            self.capacity = params["capacity"]

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = self.actor(state).detach().to("cpu").numpy()
        return action

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def get_idxes(self):
        return random.sample(range(len(self.buffer)), self.batch_size)

    def update(self, step, idxes=None):
        if step % self.update_freq != 0:
            return
        if len(self.buffer) < self.batch_size:
            return

        if idxes is None:
            idxes = self.get_idxes()
        samples = [self.buffer[idx] for idx in idxes]

        s0, a0, r1, s1 = zip(*samples)

        s0 = torch.tensor(s0, dtype=torch.float).to(self.device)
        a0 = torch.tensor(a0, dtype=torch.float).to(self.device)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1).to(self.device)
        s1 = torch.tensor(s1, dtype=torch.float).to(self.device)

        def critic_update():
            a1 = self.actor_target(s1).detach()
            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()
            y_pred = self.critic(s0, a0)

            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def actor_update():
            loss = -torch.mean(self.critic(s0, self.actor(s0)))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        critic_update()
        actor_update()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

        return idxes

    def save_model(self, config, id):
        agent_name = "agent" + str(id)
        model_path = Path(".") / config["save_path"] / config["scenario"] / "ddpg" / agent_name
        folder = os.path.exists(model_path)
        if not folder:
            os.makedirs(model_path)

        torch.save(self.actor.state_dict(), model_path / "actor.pt")
        torch.save(self.actor_target.state_dict(), model_path / "actor_target.pt")
        torch.save(self.critic.state_dict(), model_path / "critic.pt")
        torch.save(self.critic_target.state_dict(), model_path / "critic_target.pt")

    def load_model(self, config):
        model_path = Path(config["save_path"]) / config["scenario"]
        self.actor.load_state_dict(torch.load(model_path / "ddpg" / "actor.pt"))
