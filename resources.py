import subprocess
import socket
import time
import os
import numpy as np
import gymnasium
import typing

class WLEnv(gymnasium.Env):
    """ This is the main interface class for connecting WaterLily to reinforcement learning.
    
    The main idea is to offload all of the computational tasks to WL and let this class handle
    the communications and passing the data to RL agents. This should allow for as much
    generalisation and code re-use as possible without the need for re-implementing things.
    
    To avoid implementing a complex message parsing interface, all messages are padded
    to a fixed length on both Python and Julia sides. The user needs to ensure that all
    data fits within this length.
    """
    
    def __init__(self, n_state_vars=2, n_action_vars=1, msg_len=128, max_episode_steps=300):
        self.max_episode_steps = max_episode_steps
        self.msg_len = msg_len
        self.observation_space = gymnasium.spaces.Box(-1., 1., shape=(n_state_vars,), dtype=np.float64)
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(n_action_vars,), dtype=np.float64)
    
    def reset(self, seed: typing.Optional[int] = None, options: typing.Optional[dict] = None):
        """ Submit the simulation and wait until it starts to run. This will provide the
        initial observation. """
        
        episodeId = "commsTestEpisode"

        # Spawn the simulation process.
        portNumber = 8092
        self.simulation_process = subprocess.Popen(
            ["julia", "sim_02_swimmerClient.jl", episodeId, "{:d}".format(portNumber)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        # Wait a bit and open a socket.
        time.sleep(1.)
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind(('localhost', portNumber))
        # become a server socket, maximum 5 connections
        serversocket.listen(5)
        self.connection, address = serversocket.accept()
        print("Server listening")

        # Receive the initial handshake with the first observation.
        # TODO this is a bit unsafe, add try except here or some timeout.
        while True:
            time.sleep(0.5)
            buf = self.connection.recv(self.msg_len).decode("utf-8")
            if len(buf) > 0:
                print("Python got initial:", buf)
                break
            
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
                
                break
                
        observation = np.array([float(v) for v in buf.split()[1:-2]])

        # Dummy.
        info = {}
        
        # Reset step counter.
        self.iStep = 0

        return observation, info

    def step(self, action):
        # Send the action to WaterLily. This will advance the CFD time by one step
        # and a new agent state will be computed, together with the reward. These will
        # be returned via sockets.
        # TODO
        observation = None
        terminated = False
        reward = 0
        
        # Send the action.
        msg = bytes(" ".join(["{:.6e}".format(a) for a in action]) + "\n", 'UTF-8')
        padded_message = msg[:self.msg_len].ljust(self.msg_len, b'\0')
        print("Python sending:", padded_message)
        self.connection.sendall(padded_message)
       
        # Get the new observation and reward.
        # TODO this is a bit unsafe, add try except here or some timeout.
        while True:
            time.sleep(0.5)
            buf = self.connection.recv(self.msg_len).decode("utf-8")
            if len(buf) > 0:
                print("Python got:", buf)
                break
            
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
                
                break
                
        observation = np.array([float(v) for v in buf.split()[1:-2]])
        reward = float(buf.split()[-2])
        done = float(buf.split()[-1]) < 0.5
       
        # Check if the max no. steps has been reached.
        info = {}
        self.iStep += 1
        truncated = False
        if self.iStep >= self.max_episode_steps:
            terminated = True
            truncated = True
            reward = 0
        info["TimeLimit.truncated"] = truncated
        
        return observation, reward, terminated, truncated, info


class DummySwimmerAgent(object):
    def predict(self, obs, deterministic=True):
        try:
            return np.array([-np.arctan2(obs[1], obs[0]) / np.pi]), None
        except AttributeError:
            # During training, this gets given an obs and info as a tuple
            obs = obs[0]
            return np.array([-np.arctan2(obs[1], obs[0]) / np.pi]), None


def concatEpisodeData(obs, actions, reward):
    ed = {}
    ed["reward"] = reward
    for i in range(len(obs)):
        ed[f"o{i:d}"] = obs[i]
    for i in range(len(actions)):
        ed[f"a{i:d}"] = actions[i]
    return ed

