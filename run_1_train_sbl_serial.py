import matplotlib
import matplotlib.pyplot as plt
import subprocess
import socket
import time
import os
import numpy as np
import pandas
import datetime
import torch
import sb3_contrib
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

import resources

# Classical training using a single env.
agent_kwargs = {
    'learning_rate': 1e-3,
    'gamma': 0.98,
    'verbose': 0,
    'buffer_size': 25_000,
    'batch_size': 128,
    'train_freq': (1, "step"),
    "gradient_steps": 2,
}

policy_kwargs = {
    "activation_fn": torch.nn.GELU,
    "net_arch": dict(
        pi=[64, 64],
        qf=[64, 64],
    )
}

save_dir = "./trainingData_0_serialTraining"

n_evals_per_step = 3
nTrainingSteps = 50_000

try:
    # Create an env and an agent.
    env_train = resources.WLEnv("sim_02_swimmerClient.jl", os.path.join(save_dir, "episode_serialTraining"),
                                port_increment=10, n_threads=8)
    env_eval = resources.WLEnv("sim_02_swimmerClient.jl", os.path.join(save_dir, "episode_serialTraining_eval"),
                               port_increment=20, n_threads=8, verbose=True)
    agent = sb3_contrib.TQC("MlpPolicy", env_train, policy_kwargs=policy_kwargs, **agent_kwargs)
    
    # Wrap in a monitor.
    env_eval = Monitor(env_eval, os.path.join(save_dir, "eval_monitor"))

    # Create callbacks to use during training.
    model_dir = os.path.join(save_dir, "modelSaves")
    # Create callback that evaluates agent for N episodes every X training environment steps.
    eval_callback = EvalCallback(env_eval, best_model_save_path=model_dir,
                                 log_path=save_dir, eval_freq=250,
                                 n_eval_episodes=n_evals_per_step, deterministic=True,
                                 render=False, verbose=1)
    # Create a callback for saving models every now and again irrespective of everything.
    save_callback = resources.PeriodicallySaveModelCallback(check_freq=100, log_dir=model_dir, verbose=1)
    # Stack.
    callback = CallbackList([eval_callback, save_callback])

    # Train
    time_start = datetime.datetime.now()
    agent.learn(total_timesteps=nTrainingSteps, log_interval=10, progress_bar=False, callback=callback)
    training_time = (datetime.datetime.now() - time_start).total_seconds()
    print("Learning took {:.0f} seconds".format(training_time))

    # Save the final agent.
    save_path = os.path.join(model_dir, "model_sbl_serial_final")
    agent.save(save_path)

finally:
    # Make sure nothing gets left to its own devices.
    env_train.close()

# Check the training.
matplotlib.rc("font", **{"family": "serif", "weight": "normal", "size": 15})
matplotlib.rcParams["figure.figsize"] = (9, 8)

eval_data, eval_summary = resources.read_monitor(
    os.path.join(save_dir, "eval_monitor.monitor.csv"), n_evals_per_step)

fig, ax = resources.plot_training(eval_data, eval_summary)

plt.show()


