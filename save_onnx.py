from pathlib import Path

import torch
import netron

from algorithm.multi.ddpg import Actor, Critic
from algorithm.utils import get_multi_agent
from config import config, params
from make_env import make_env

import numpy as np

algorithm = "ddpg"

env = make_env(config["scenario"])
state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n

state = env.reset()
agent = get_multi_agent(algorithm)(state_dim, action_dim, params)
action = agent.act(state)

model_path = Path(config["save_path"]) / config["scenario"]

is_actor = False
if is_actor:
    input_state = torch.tensor(state, dtype=torch.float)
    torch.onnx.export(agent.actor, input_state, "output/actor.onnx", verbose=True,
                      input_names=["state", "layer1", "layer2", "layer3"],
                      output_names=["action"])
    netron.start("output/actor.onnx")
else:
    input_state = torch.tensor(state[0], dtype=torch.float)
    input_action = torch.tensor(action[0], dtype=torch.float)
    torch.onnx.export(agent.critic, (input_state, input_action), "output/actor.onnx", verbose=True,
                      input_names=["state", "action", "layer2", "layer3"],
                      output_names=["Q value"])
    netron.start("output/actor.onnx")

