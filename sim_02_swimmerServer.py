import subprocess
import socket
import time
import os
import numpy as np

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

obs, info = env_eval.reset()
episodeData = [resources.concatEpisodeData(obs, np.zeros(len(env_eval.action_space.sample())), 0.)]

for i in range(nStepsMax):
    action, _states = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env_eval.step(action)
    episodeData.append(resources.concatEpisodeData(obs, action, reward))
    if terminated:
        break

episodeData = pandas.DataFrame(episodeData)


"""
        try:
            # Keep running the simulation.
            while True:
                # Chill for a bit.
                time.sleep(0.5)
                print("I'm alive")
                
                # Check if the process has finished
                return_code = self.simulation_process.poll()
                if return_code is not None:
                    print(f"Simulation finished with exit code {return_code}")
                    
                    # Capture all the outputs when the process finishes
                    stdout_data, stderr_data = self.simulation_process.communicate()
                    
                    # TODO dumpt this to a log.
                    print("Final output:")
                    print(stdout_data.decode())
                    print(stderr_data.decode())

                    # Close the socket.
                    self.connection.close()
                
                    break
                        
                # Get the new observation and reward.
                buf = self.connection.recv(self.msg_len).decode("utf-8")
                if len(buf) > 0:
                    print("Python got:", buf)
                    
        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            # Cleanup: Close TCP connection and terminate the subprocess if still running
            print("Cleaning up...")
            
            if self.simulation_process.poll() is None:
                print("Terminating simulation process...")
                self.simulation_process.terminate()
                self.simulation_process.wait()
                


stdout_data, stderr_data = simulation_process.communicate()
print(stdout_data.decode())
"""
