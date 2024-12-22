import subprocess
import socket
import time
import os
import numpy as np
import gymnasium
import typing

# ========================================================
# Generic functions and classes for WL and RL interfacing.
# ========================================================

def isSimRunning(simulation_process, verbose=True):
    return_code = simulation_process.poll()
    if return_code is not None:
        if verbose:
            print(f"Simulation finished with exit code {return_code}")
        
        # Capture all the outputs when the process finishes
        stdout_data, stderr_data = simulation_process.communicate()

        if verbose:        
            print("Final output:")
            print(stdout_data.decode())
            print(stderr_data.decode())

        return False, stdout_data.decode() + stderr_data.decode()
    
    else:
        return True, None


def send(data, connection, msg_len=124):
    msg = bytes(" ".join(["{:.6e}".format(value) for value in data]) + "\n", 'UTF-8')
    padded_message = msg[:msg_len].ljust(msg_len, b'\0')
    connection.sendall(padded_message)


def receive(connection, simulation_process, logdir, msg_len=128, delay=0.1):
    # Receive the initial handshake with the first observation.
    while True:
        time.sleep(delay)

        # Return the message if all's fine.        
        buf = connection.recv(msg_len).decode("utf-8")
        if len(buf) > 0:
            break
        
        # Check the sim status.
        simOk, log = isSimRunning(simulation_process)
        
        if not simOk:
            # log the simulation output and let the calling
            # code know that the simulation is no longer running. This will not be
            # called on a regular pass as it's meant for debugging only. Use a regular
            # log on the Julia side for regular bookkeeping.
            with open(os.path.join(logdir, "log_server.txt"), "w") as logfile:
                logfile.write(log)

            return simOk, log
        
    return True, buf
        

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
        self.observation_space = gymnasium.spaces.Box(-1., 1., shape=(n_state_vars,), dtype=np.float64)
        self.n_action_vars = n_action_vars
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(n_action_vars,), dtype=np.float64)
        self.simulation_process = None
        
        # Length of the message after formatting values to text and padding. Should
        # be long enough to enclose the entire data packet, otherwise data loss will occur.
        self.msg_len = msg_len
        
        # Delay in seconds for giving Julia time to finish a simulation in the middle
        # of a time step. Should be considerably longer than the wall time per time step
        # but not infite.
        self.kill_time_delay = 5.
        
        # Delay in seconds between starting a simulation process and opening the socket.
        # Shouldn't really change much.
        self.start_time_delay = 1.
    
    def killSim(self):
        if self.simulation_process is not None:
            if self.simulation_process.poll() is None:
                # Create a killfile.
                print("Terminate, terminate!")
                with open(os.path.join(self.episode_dir, "killfile"), "w") as f:
                    f.write("Terminate, terminate!")
                    
                # Write some random data to the socket to prevent it locking up.
                send(np.zeros(self.n_action_vars), self.connection, msg_len=self.msg_len)
        
                # Wait a bit to give Julia time to wrap things up in an orderly fashion.
                time.sleep(self.kill_time_delay)
                
                # Force kill.
                self.simulation_process.terminate()
                self.simulation_process.wait()
                
                # Grab output and log.
                stdout_data, stderr_data = self.simulation_process.communicate()
                log = stdout_data.decode() + stderr_data.decode()
                with open(os.path.join(self.episode_dir, "log_server.txt"), "w") as logfile:
                    logfile.write(log)
                    
                
                print("Final output:")
                print(log)
                    
                return True
        return False

    def reset(self, seed: typing.Optional[int] = None, options: typing.Optional[dict] = None):
        """ Submit the simulation and wait until it starts to run. This will provide the
        initial observation. """

        # TODO change for subsequent episodes.        
        self.port_number = 8092
        self.episode_dir = "episode_commsTestEpisode"
        self.sim_exe = "sim_02_swimmerClient.jl"
        
        # Check if a previous process needs to be killed.
        self.killSim()
        
        # Spawn the simulation process.
        self.simulation_process = subprocess.Popen(
            ["julia", self.sim_exe, self.episode_dir, "{:d}".format(self.port_number)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )

        # Wait a bit and open a socket.
        time.sleep(self.start_time_delay)
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind(('localhost', self.port_number))
        # become a server socket, maximum 5 connections
        serversocket.listen(5)
        self.connection, address = serversocket.accept()
        print(f"Server listening on port {self.port_number}")

        # Receive the initial handshake with the first observation.
        simOk, buf = receive(self.connection, self.simulation_process, self.episode_dir, msg_len=self.msg_len)
        
        # This shouldn't happen...
        if not simOk:
            raise RuntimeError("Simulation did not start correctly, check the log! Usually this means an issue on the Julia side.")

        # Convert the buffer to the initial observation. Last two entries are reward and done.                
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
        
        # Send the action.
        send(action, self.connection, msg_len=self.msg_len)
       
        # Get the new observation and reward.
        simOk, buf = receive(self.connection, self.simulation_process, self.episode_dir, msg_len=self.msg_len)
        
        # Split into components from the vector message.                
        if simOk:
            observation = np.array([float(v) for v in buf.split()[1:-2]])
            reward = float(buf.split()[-2])
            done = float(buf.split()[-1]) > 0.5
        else:
            # This shouldn't happen...
            print("Bad sim, bad!")
            observation = None
            reward = 0
            done = True
        
        # Check if the max no. steps has been reached.
        info = {}
        self.iStep += 1
        truncated = False
        if self.iStep >= self.max_episode_steps:
            done = True
            truncated = True
        info["TimeLimit.truncated"] = truncated
        
        return observation, reward, done, truncated, info

    def close(self):
        self.killSim()


def concatEpisodeData(obs, actions, reward):
    ed = {}
    ed["reward"] = reward
    for i in range(len(obs)):
        ed[f"o{i:d}"] = obs[i]
    for i in range(len(actions)):
        ed[f"a{i:d}"] = actions[i]
    return ed


# =============================
# Specific to the swimmer case
# =============================


class DummySwimmerAgent(object):
    """ Sets the heading directly towards the target, assuming the first two
    values of the observations vector are unit vector components pointing towards
    the target. """
    def predict(self, obs, deterministic=True):
        try:
            return np.array([np.arctan2(obs[1], obs[0]) / np.pi]), None
        except AttributeError:
            # During training, this gets given an obs and info as a tuple
            obs = obs[0]
            return np.array([np.arctan2(obs[1], obs[0]) / np.pi]), None



