import subprocess
import socket
import time
import os
import numpy as np
import pandas
import datetime
import torch

import sb3_contrib
import stable_baselines3
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer

import resources


# Training using asynchronous data collection (patent pending).


class AsyncReplayBuffer(ReplayBuffer):
    """ Buffer that allows data to be added in an arbitrary fashion, allowing
    asynchronous collection of action-state pairs. """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: typing.Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs,
                         optimize_memory_usage=False, handle_timeout_termination=True)

    def add_batch(self, observations, actions, rewards, next_observations, dones, infos):
        """
        Add a batch of transitions to the buffer.
        observations, actions, rewards, next_observations, dones are arrays of shape (batch_size, ...)

        Note that the super().add interface is as follows:

            def add(
                self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
            ) -> None:
        """
        for i in range(len(observations)):
            self.add(
                obs=observations[i],
                next_obs=next_observations[i],
                action=actions[i],
                reward=rewards[i],
                done=dones[i],
                infos=[infos[i]]
            )

nParallelJobs = 5
buffer_size = 10_000
n_steps_per_buffer_update = 100
n_iterations = 100
batch_size = 64
n_batch_to_train_on = 10
modelName = "test"
model_dir = "./models"

# Create the vectorised env.
envs = []
for i in range(nParallelJobs):
    envs.append(
        resources.WLEnv("sim_02_swimmerClient.jl", f"episode_serialTraining_rank_{i:d}", port_increment=i))

# Instantiate an agent.
agent = sb3_contrib.TQC("MlpPolicy", envs[0], policy_kwargs=policy_kwargs, **agent_kwargs)

# Dummy logger with minimal functionality
agent_logger = stable_baselines3.common.logger.configure(model_dir, ["stdout"])
agent.set_logger(agent_logger)

# Create a buffer inside the agent.
agent.replay_buffer = AsyncReplayBuffer(buffer_size, envs[0].observation_space,
                                        envs[0].action_space, agent.device, n_envs=1)

# Crank the sausage machine.
time_start = datetime.datetime.now()

# Start the simulations.
obses = []
for env in envs:
    obs, info = env.reset(nonblocking=True)
    obses.append(obs)

# Run the training. Use while loop to allow for slow start-up of the simulations
# and ensure that the specified no. calls to the buffer actually happens.
iTrainingStep = 0
while iTrainingStep < n_iterations:

    # Run the environments until the required no. time steps has been collected.
    obs_list, actions_list, rewards_list, next_obs_list, dones_list, infos_list = \
        [], [], [], [], [], []
    while len(obs_list) < n_steps_per_buffer_update:
        for iEnv in range(len(envs)):
            # Check if the env is running. If not, retry the initial handshake.
            if obses[iEnv] is None:
                obs, info = envs[iEnv].retryInitialHandshake()
                obses[iEnv] = obs

            if obses[iEnv] is not None:
                # Make a step and get the observation.
                action, _ = agent.predict(obses[iEnv], deterministic=False)
                next_obs, reward, done, truncated, info = envs[iEnv].step(action, blocking=False)
                # Non-blocking comms return None state if the data is not yet valid.
                if next_obs is None:
                    # Go to the next env.
                    continue
                else:
                    # Store the data.
                    obs_list.append(obses[iEnv])
                    actions_list.append(action)
                    rewards_list.append(reward)
                    next_obs_list.append(next_obs)
                    dones_list.append(done)
                    infos_list.append(info)
                    # Set state for the next step.
                    obses[iEnv] = next_obs

                if done:
                    # Run a new simulation and wait for the initial handshake. If this fails,
                    # the obs will be None and will need to be retried later.
                    obs, info = envs[iEnv].reset(nonblocking=True)
                    obses[iEnv] = obs

    # Move the data to the buffer.
    agent.replay_buffer.add_batch(obs_list, actions_list, rewards_list, next_obs_list, dones_list, infos_list)

    # Train the agent.
    if agent.replay_buffer.size() > batch_size:
        iTrainingStep += 1
        print("Trainig step", iTrainingStep)
        for _ in range(n_batch_to_train_on):
            agent.train(gradient_steps=1, batch_size=batch_size)

    # Save the agent.
    save_path = os.path.join(model_dir, "model_{:s}_step_{:d}".format(modelName, iTrainingStep))
    agent.save(save_path)

# Close the envs
for env in envs:
    env.app_link.stop_app()
    env.close()

time_end = datetime.datetime.now()
print("Learning took {:.0f} seconds".format((time_end-time_start).total_seconds()))

