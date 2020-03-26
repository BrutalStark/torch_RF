from algorithm.multi.ddpg import ddpgAgent


class maddpg:
    def __init__(self, state_dim_n, action_dim_n, params):
        self.algorithm = params["algorithm"]

        self.update_freq = params["update_freq"]
        self.batch_size = params["batch_size"]

        self.agent_num = params["agent_num"]
        self.agent_n = [ddpgAgent(state_dim, action_dim, params)
                        for state_dim, action_dim in zip(state_dim_n, action_dim_n)]

    def act(self, state_n):
        return [agent.act(state) for agent, state in zip(self.agent_n, state_n)]

    def put(self, s0_n, a0_n, r1_n, s1_n):
        for agent, s0, a0, r1, s1 in zip(self.agent_n, s0_n, a0_n, r1_n, s1_n):
            agent.put(s0, a0, r1, s1)

    def update(self, step_num):
        if step_num % self.update_freq != 0:
            return
        if len(self.agent_n[0].buffer) < self.batch_size:
            return

        if self.algorithm is "ddpg":
            idxes = self.agent_n[0].get_idxes()
            for agent in self.agent_n:
                agent.update(step_num, idxes)

    def save_model(self, config):
        for n in range(self.agent_num):
            agent = self.agent_n[n]
            agent.save_model(config, n)
