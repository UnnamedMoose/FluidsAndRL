import socket
import time
import selectors
import os
import numpy as np
import pandas
import datetime
import subprocess

import resources

#def send(data, connection, msg_len=128):
#    msg = bytes(" ".join(["{:.6e}".format(value) for value in data]) + "\n", 'UTF-8')
#    padded_message = msg[:msg_len].ljust(msg_len, b'\0')
#    connection.sendall(padded_message)


def accept_connection(server_sock, ports):
    conn, addr = server_sock.accept()
    laddr = conn.getsockname()
    iWorker = ports.index(laddr[1])
    conn.setblocking(False)
    print(f"Non-blocking connection from port {laddr}")
    sel.register(conn, selectors.EVENT_READ, handle_client)
    return iWorker, None, None, None, False


def handle_client(conn, ports):
    try:
        data = conn.recv(128)
        if data:
            laddr = conn.getsockname()
            raddr = conn.getpeername()
            iWorker = ports.index(laddr[1])

            print(f"Received from worker {iWorker:d} on {laddr}: {data.decode()}")
            
            # This effectively circumnavigates the step() method in the env
            # to make sure the same routines can be called through callbacks.
            vals = data.decode().split()
            vals = [int(vals[0])] + [float(v) for v in vals[1:]]
            
            obs = np.array(vals[1:-2])
            reward = float(vals[-2])
            done = vals[-1] > 0.5
            
            # TODO call agent .predict() method here
            action = np.array([10., 20., 30.])
            resources.send(action, conn)
            
            return iWorker, obs, reward, action, done
            
        else:
            laddr = conn.getsockname()
            iWorker = ports.index(laddr[1])
            print(f"Client {iWorker:d} on {laddr} disconnected")
            sel.unregister(conn)
            conn.close()
            return iWorker, None, None, None, True

    except ConnectionResetError:
        print("Connection reset by peer")
        sel.unregister(conn)
        conn.close()

# TODO replace with the env routine
#def kill_process(simulation_process):
#    if simulation_process.poll() is None:
        # Create a killfile.
#                with open(os.path.join(self.episode_dir, "killfile"), "w") as f:
#                    f.write("Terminate, terminate!")
            
        # Write some random data to the socket to prevent it locking up.
        # (Guess how I found out this was necessary?)
#                send(np.zeros(self.n_action_vars), self.connection, msg_len=self.msg_len)

        # Wait a bit to give Julia time to wrap things up in an orderly fashion.
#                time.sleep(self.kill_time_delay)
        
        # Force kill.
#        simulation_process.terminate()
#        simulation_process.wait()


nParallelJobs = 3
nStepsToCollect = 50

# Create the parallel envs.
sel = selectors.DefaultSelector()
envs = []
ports = []
for i in range(nParallelJobs):
    envs.append(resources.WLEnv(
        "sim_00_dummySim.jl",
        f"episode_parallelTraining_test_worker_{i:d}", 
        n_state_vars=3,
        n_action_vars=3,
        port_increment=i+1234,
        verbose=True,
    ))
    envs[-1].openPort(blocking=False)
    
    # Register the callback that accepts data from each client and executes
    # the parsing instructions.
    sel.register(envs[-1].server_sock, selectors.EVENT_READ, accept_connection)
    ports.append(envs[-1].port_number)

print("Servers are listening on ports:", ports)

# Main loop to submit jobs, poll events, and get data back.
collectedData = []
isRunning = [False for _ in range(len(ports))]
#simulation_processes = [None for _ in range(len(ports))]
jobIds = [-1 for _ in range(len(ports))]
episodeCount = 0

# Collect a bit more data to make sure episodes with one step etc. don't mess up
# the process.
while len(collectedData) < nStepsToCollect+episodeCount+1:
    # This shouldn't be needed but it's better not to get sysadmins angry with
    # loads of comms, especially between the compute and login nodes.
    time.sleep(0.1)
    
    # Check if all jobs slots are active. Submit a new simulation if not.
    for iWorker, port in enumerate(ports):
        if not isRunning[iWorker]:
            # Star the job.
            episodeCount += 1
            jobIds[iWorker] = episodeCount
            envs[iWorker].reset(asynchronous=True)
            isRunning[iWorker] = True
            
            print(f"Started sim {episodeCount:d} using slot {iWorker:d}")
            
            # Wait some time to artificially desynchronise the processes, emulating what
            # working with CFD might look like.
            # TODO temp
            time.sleep(0.5 + np.random.rand()*2)

    # Grab the observations as they come in.
    events = sel.select(timeout=None)
    for key, mask in events:
        callback = key.data
        iWorker, obs, reward, action, done = callback(key.fileobj, ports)
        print(f"  {iWorker:d} returned:", done, obs, reward, action)
        
        if not done and obs is not None:
            # Regular time step. Keep data.
            now = datetime.datetime.now()
            collectedData.append(np.concatenate(
                [[jobIds[iWorker], iWorker, now], obs, action, [reward]]))
        
        elif not done and obs is None:
            # First handshake with no valid data. Ignore.
            pass

        else:
            # Job terminated and worker disconnected.
            isRunning[iWorker] = False
            
            # Ensure proper clean up. This shouldn't be necessary here as
            # simulation_process.poll() will not be None at this point, but you
            # never know.
            envs[iWorker].killSim(brutal=True)
            #kill_process(envs[iWorker].simulation_process)
            print(f"Worker {iWorker:d} killed the subprocess properly")

# Terminate running jobs.
for iWorker in range(len(ports)):
    if isRunning[iWorker]:
        envs[iWorker].killSim(brutal=True)
        #kill_process(envs[iWorker].simulation_process)
        print(f"Worker {iWorker:d} killed the subprocess properly")

# Get data into orderly format.
x_obs = [f"obs_{i:d}" for i in range(envs[0].observation_space.shape[0])]
x_act = [f"act_{i:d}" for i in range(envs[0].action_space.shape[0])]
collectedData = pandas.DataFrame(data=np.array(collectedData),
    columns=["iEp", "iWorker", "time"] + x_obs + x_act + ["reward"]).sort_values(["iEp", "time"])

# Note: the way the simulations are run, the action at each index is the action used
# to move to the next row but reward is the reward for the previous transition.
# Need to unpack this all into what stable baselines expect.

parsedData = []
for iEp in collectedData["iEp"].unique():
    df = collectedData[collectedData["iEp"] == iEp]
    if len(df) < 2:
        continue
    
    # Time per time step in seconds.
    t = (df["time"].values[1:] - df["time"].values[:-1]).astype(float)[:, np.newaxis] / 1e9

    # State-action-reward transition data.
    obs = df[x_obs].values[:-1]
    action = df[x_act].values[:-1]
    next_obs = df[x_obs].values[1:]
    reward = df["reward"].values[1:][:, np.newaxis]
    
    # Stack.
    x_obs_n = [f"obs_next_{i:d}" for i in range(len(x_obs))]
    parsedData.append(pandas.DataFrame(
        data=np.column_stack([obs, action, next_obs, reward, t]),
        columns=x_obs+x_act+x_obs_n+["reward", "time"]))
    parsedData[-1]["iEp"] = iEp

collectedData = pandas.concat(parsedData[:nStepsToCollect]).reset_index(drop=True)

# TODO merge with the training part in v0

