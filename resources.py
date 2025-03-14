import subprocess
import socket
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas
import typing
import gymnasium
from stable_baselines3.common.callbacks import BaseCallback

# ========================================================
# Generic functions and classes for WL and RL interfacing.
# ========================================================

def isSimRunning(simulation_process, verbose=False):
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


def send(data, connection, msg_len=128):
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
    
    def __init__(self, sim_exe, episode_dir, n_state_vars=4, n_action_vars=1, msg_len=128, kill_time_delay=5.,
            max_episode_steps=300, base_port_number=8089, port_increment=0, n_threads=None, verbose=False):
        self.max_episode_steps = max_episode_steps
        self.observation_space = gymnasium.spaces.Box(-1., 1., shape=(n_state_vars,), dtype=np.float64)
        self.n_action_vars = n_action_vars
        self.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(n_action_vars,), dtype=np.float64)
        
        # Additional printout for debugging.
        self.verbose = verbose
        
        # This is the secret tool that we will need later.
        self.simulation_process = None
        self.server_sock = None
        self.connection = None
        
        # Julia sim to run.
        self.sim_exe = sim_exe
        
        # Parallel running control of the Julia sim.
        self.n_threads = n_threads
        
        # Increment can be used to have several simulations run concurrently.
        self.port_number = base_port_number + port_increment
        # Note that episode dir will be appendded with total episode count.
        self.base_episode_dir = episode_dir
        self.n_episodes = 0
                
        # Length of the message after formatting values to text and padding. Should
        # be long enough to enclose the entire data packet, otherwise data loss will occur.
        self.msg_len = msg_len
        
        # Delay in seconds for giving Julia time to finish a simulation in the middle
        # of a time step. Should be considerably longer than the wall time per time step
        # but not infite.
        self.kill_time_delay = kill_time_delay
        
        # Delay in seconds between starting a simulation process and opening the socket.
        # Shouldn't really change much.
        self.start_time_delay = 1.
    
    def killSim(self, brutal=False):
        if self.simulation_process is not None:
            if self.simulation_process.poll() is None:
                if not brutal:
                    # Create a killfile.
                    with open(os.path.join(self.episode_dir, "killfile"), "w") as f:
                        f.write("Terminate, terminate!")
                    if self.verbose: print("Killfile created.")
                        
                    # Write some random data to the socket to prevent it locking up.
                    # (Guess how I found out this was necessary?)
                    send(np.zeros(self.n_action_vars), self.connection, msg_len=self.msg_len)
                    if self.verbose: print("Random data written.")
            
                    # Wait a bit to give Julia time to wrap things up in an orderly fashion.
                    time.sleep(self.kill_time_delay)
                    if self.verbose: print("Kill wait over.")
                
                # Force kill.
                self.simulation_process.terminate()  # Send SIGTERM - soft kill
                try:
                    self.simulation_process.wait(timeout=self.kill_time_delay)  # Wait for the process to exit
                except subprocess.TimeoutExpired:
                    if self.verbose: print("Process did not terminate in time, force killing...")
                    self.simulation_process.kill()  # Send SIGKILL (hard kill)
                    self.simulation_process.wait()  # Ensure cleanup
                if self.verbose: print("Process termindated.")
                
                # Grab output and log it.
                if not brutal:
                    _, log = isSimRunning(self.simulation_process)
                    with open(os.path.join(self.episode_dir, "log_server.txt"), "w") as logfile:
                        logfile.write(log)
                    if self.verbose: print("Post-kill log written.")
                
                # Close the socket. It could be none in asynchronous mode. In that case,
                # the connections get handled externally.
                if self.connection is not None:
                    self.connection.close()
                    if self.verbose: print("Socket closed.")
                    
                return True
        
        return False

    def openPort(self, blocking=True):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Enable SO_REUSEADDR to reuse the port
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(('localhost', self.port_number))
        # become a server socket, maximum 5 connections. This is overkill but meh.
        self.server_sock.listen(5)
        if not blocking:
            self.server_sock.setblocking(False)
        if self.verbose: print("Port opened.")
        
    def reset(self, seed: typing.Optional[int] = None, options: typing.Optional[dict] = None, asynchronous: typing.Optional[bool] = False):
        """ Submit the simulation and wait until it starts to run. This will provide the
        initial observation. """
        
        if self.verbose: print("---\nReset started.")
        
        # Set the random seed.
        super().reset(seed=seed)

        # Check if a previous process needs to be killed.
        self.killSim()
        if self.verbose: print("Old sim killed.")

        # Set the new count and make a unique path.
        self.n_episodes += 1
        self.episode_dir = self.base_episode_dir+f"_ep_{self.n_episodes:d}"
                
        # Spawn the simulation process.
        if self.n_threads is None:
            paropt = "-t 1"
        elif type(self.n_threads) is str:
            paropt = "-t "+self.n_threads
        else:
            paropt = f"-t {self.n_threads:d}"

        cmd = [
            "julia",
            paropt,
            self.sim_exe,
            self.episode_dir,
            "{:d}".format(self.port_number)
        ]
        self.simulation_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        if self.verbose: print("New sim process created:", " ".join(cmd))

        # Wait a bit and open a socket.
        time.sleep(self.start_time_delay)
#        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Enable SO_REUSEADDR to reuse the port
#        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#        self.server_sock.bind(('localhost', self.port_number))
        # become a server socket, maximum 5 connections
#        self.server_sock.listen(5)

        if asynchronous:
            if self.verbose: print("Reset done for asynchronous mode.\n---")
            # Return here and let the overarching code deal with things.
            return None, {}
        else:
            self.openPort()
            if self.verbose: print("Old sim killed.")

        # In serial mode, wait for proper start up.
        self.connection, address = self.server_sock.accept()
        if self.verbose: print(f"Server listening on port {self.port_number}")

        # Receive the initial handshake with the first observation.
        simOk, buf = receive(self.connection, self.simulation_process, self.episode_dir, msg_len=self.msg_len)
        if self.verbose: print("Initial receive successful.")
        
        # This shouldn't happen...
        if not simOk:
            raise RuntimeError("Simulation did not start correctly, check the log! Usually this means an issue on the Julia side.")

        # Convert the buffer to the initial observation. Last two entries are reward and done.                
        observation = np.array([float(v) for v in buf.split()[1:-2]])

        # Dummy.
        info = {}
        
        # Reset step counter.
        self.iStep = 0
        
        if self.verbose: print("Reset done.\n---")

        return observation, info

    def step(self, action):
        # Send the action to WaterLily. This will advance the CFD time by one step
        # and a new agent state will be computed, together with the reward. These will
        # be returned via sockets.
        
        if self.verbose: print("---\nStep started.")
        
        # Send the action.
        send(action, self.connection, msg_len=self.msg_len)
        if self.verbose: print("Actions sent.")
       
        # Get the new observation and reward.
        simOk, buf = receive(self.connection, self.simulation_process, self.episode_dir, msg_len=self.msg_len)
        if self.verbose: print("State received.")
        
        # Split into components from the vector message.                
        if simOk:
            observation = np.array([float(v) for v in buf.split()[1:-2]])
            reward = float(buf.split()[-2])
            done = float(buf.split()[-1]) > 0.5
        else:
            # This shouldn't happen but it does during the last time step when training with SBL.
            #print("Bad sim, bad!")
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
        
        if self.verbose: print("Step done. Sim done:", done, "\n---")
        
        return observation, reward, done, truncated, info

    def close(self):
        self.killSim()


# ===
# Shorthands and tools for RL training
# ===


def concatEpisodeData(obs, actions, reward):
    ed = {}
    ed["reward"] = reward
    for i in range(len(obs)):
        ed[f"o{i:d}"] = obs[i]
    for i in range(len(actions)):
        ed[f"a{i:d}"] = actions[i]
    return ed


def evaluate_agent(agent, env, num_episodes=1, num_steps=None, deterministic=True,
                   render=False, seed=None, reset_kwargs={}):
    # This function will only work for a serial environment
    frames = []
    histories = []
    all_episode_rewards = []
    for iEp in range(num_episodes):

        # Prepare for a new episode.
        obs, info = env.reset(seed=seed, **reset_kwargs)
        
        # Run until termination if not specified.
        if num_steps is None:
            num_steps = 1000000
        
        # Prepare storage container for the time history.
        episodeData = [concatEpisodeData(obs, np.zeros(len(env.action_space.sample())), 0.)]

        # Loop over all the needed steps.
        episode_rewards = []
        frames.append([])
        terminated = False
        for i in range(num_steps):
            # Querry the agent's policy for an action.
            # _states are only useful when using LSTM policies
            action, _states = agent.predict(obs, deterministic=deterministic)
            
            # Take a single step.
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check if done.
            if terminated:
                break
            
            # Keep reward.
            episode_rewards.append(reward)
            
            # Make a pretty picture, maybe.
            if render:
                frames[-1].append(env.render(mode="rgb_array"))
            
            # Keep state transition as long as obs is not None (for terminated episodes).
            episodeData.append(concatEpisodeData(obs, action, reward))

        all_episode_rewards.append(episode_rewards[-1])
        histories.append(pandas.DataFrame(episodeData))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("  Mean reward:", mean_episode_reward)

    return mean_episode_reward, all_episode_rewards, histories, frames


class PeriodicallySaveModelCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(PeriodicallySaveModelCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir

    def _init_callback(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.save_path = os.path.join(
                self.log_dir, "model_step_{:d}".format(self.n_calls))
            self.model.save(self.save_path)

        return True


def read_monitor(filename, n_evals_per_step=1):
    # TODO if this is called while evaluation is running, problems will ensure
    eval_data = pandas.read_csv(filename, header=1)
    eval_data["iEpisode"] = np.concatenate([[i]*n_evals_per_step for i in range(eval_data.shape[0]//n_evals_per_step)])[:len(eval_data)]
    eval_summary = pandas.DataFrame({
        "iEpisode": eval_data["iEpisode"].unique(),
        "reward_mean": [eval_data.loc[eval_data["iEpisode"] == i, "r"].mean() for i in eval_data["iEpisode"].unique()],
        "reward_min": [eval_data.loc[eval_data["iEpisode"] == i, "r"].min() for i in eval_data["iEpisode"].unique()],
        "reward_max": [eval_data.loc[eval_data["iEpisode"] == i, "r"].max() for i in eval_data["iEpisode"].unique()],
    })
    return eval_data, eval_summary
    
# ===
# Just some plotting.
# ===


def makeNiceAxes(ax, xlab=None, ylab=None):
    ax.tick_params(axis='both', reset=False, which='both', length=5, width=2)
    ax.tick_params(axis='y', direction='out', which="both")
    ax.tick_params(axis='x', direction='out', which="both")
    for spine in ['top', 'right','bottom','left']:
        ax.spines[spine].set_linewidth(2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)


def niceFig(xlab=None, ylab=None, figsize=None, nrows=1, ncols=1):
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
    if (nrows == 1) and (ncols == 1):
        makeNiceAxes(ax, xlab, ylab)
    else:
        for axx in ax:
            try:
                makeNiceAxes(axx, xlab, ylab)
            except AttributeError:
                for axxx in axx:
                    makeNiceAxes(axxx, xlab, ylab)
    return fig, ax


def plot_training(eval_data, eval_summary, n_rolling=5):
    fig, ax = niceFig("Evaluation step", "Reward")
    ax.plot(eval_data["iEpisode"], eval_data["r"], "ko", ms=7, label="Individual episodes")
    ax.fill_between(eval_summary["iEpisode"], eval_summary["reward_min"], eval_summary["reward_max"], color="k", alpha=0.2, zorder=-100)
    ax.plot(eval_summary["iEpisode"], eval_summary["reward_mean"], "r:", ms=7, lw=2, alpha=1, label="Batch average")
    ax.plot(eval_summary["iEpisode"], eval_summary["reward_mean"].rolling(n_rolling).mean(), "r-", lw=4, alpha=0.6, label="Rolling average")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3)
    return fig, ax

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



