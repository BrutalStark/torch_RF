from algorithm.multi.ddpg import ddpgAgent
from algorithm.multi.maddpg import maddpg


def get_multi_agent(agent_name):
    if agent_name == "ddpg":
        return ddpgAgent
    if agent_name == "maddpg":
        return maddpg