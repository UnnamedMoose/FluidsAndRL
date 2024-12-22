import subprocess
import socket
import time
import os
import numpy as np
import pandas

import resources

# Instantiate the env and a deterministic agent.
env_eval = resources.WLEnv()
agent = resources.DummySwimmerAgent()

# Run a trial evaluation
nStepsMax = 20

try:
    obs, info = env_eval.reset()
    episodeData = [resources.concatEpisodeData(obs, np.zeros(len(env_eval.action_space.sample())), 0.)]

    for i in range(nStepsMax):
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_eval.step(action)
        episodeData.append(resources.concatEpisodeData(obs, action, reward))
        if terminated:
            break

    episodeData = pandas.DataFrame(episodeData)

finally:
    env_eval.close()

