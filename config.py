import torch

config = {
    "scenario": "simple",
    "save_path": "./output",
    "algorithm": "ddpg",
    # training
    "save_episodes": 1000,
    "log_episodes": 1000,
    "episodes": 25001,
    "steps_max": 25,
    # testing
    "test_step": 100,
}
params = {
    "device": "cpu",
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # update frequency
    "update_freq": 100,
    # train or test
    "mode": "train",
    # learning rate of actor and critic
    "hidden_dim": 256,
    "lr_actor": 1e-3,
    "lr_critic": 1e-3,
    # reinforcement learning params
    'gamma': 0.99,
    'tau': 0.02,
    # replay buffer
    'capacity': 1e6,
    'batch_size': 4096,
}
is_action_discrete = {
    "ddpg": False
}