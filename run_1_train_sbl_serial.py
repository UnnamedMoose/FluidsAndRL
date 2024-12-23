import subprocess
import socket
import time
import os
import numpy as np
import pandas
import datetime
import torch
import sb3_contrib

import resources

# Classical training using a single env.

agent_kwargs = {
    'learning_rate': 1e-2,
    'gamma': 0.98,
    'verbose': 0,
    'buffer_size': 5_000,
    'batch_size': 32,
    'train_freq': (1, "step"),
    "gradient_steps": 1,
}

policy_kwargs = {
    "activation_fn": torch.nn.GELU,
    "net_arch": dict(
        pi=[32, 32],
        qf=[32, 32],
    )
}

model_dir = "./models"

nTrainingSteps = 10_000

try:
    # Create an env and an agent.
    env_train = resources.WLEnv("sim_02_swimmerClient.jl", "episode_serialTraining")
    agent = sb3_contrib.TQC("MlpPolicy", env_train, policy_kwargs=policy_kwargs, **agent_kwargs)

    # Train
    time_start = datetime.datetime.now()
    agent.learn(total_timesteps=nTrainingSteps, progress_bar=True)
    training_time = (datetime.datetime.now() - time_start).total_seconds()
    print("Learning took {:.0f} seconds".format(training_time))

    # Save the agent.
    save_path = os.path.join(model_dir, "model_sbl_serial")
    agent.save(save_path)

finally:
    # Make sure nothing gets left to its own devices.
    env_train.close()

