import subprocess
import socket
import time
import os
import numpy as np
import pandas

import sb3_contrib
import stable_baselines3
import gymnasium
import typing
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer

import resources

# Instantiate the env and a deterministic agent.
env_eval = resources.WLEnv()
agent = resources.DummySwimmerAgent()

# Run a trial evaluation
nStepsMax = 200

#try:
obs, info = env_eval.reset()
episodeData = [resources.concatEpisodeData(obs, np.zeros(len(env_eval.action_space.sample())), 0.)]

for i in range(nStepsMax):
    action, _states = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_eval.step(action)
    episodeData.append(resources.concatEpisodeData(obs, action, reward))
    if terminated:
        break

episodeData = pandas.DataFrame(episodeData)

#finally:
env_eval.close()

