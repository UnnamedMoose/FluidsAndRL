import socket
import time
import selectors
import os
import numpy as np
import pandas
import datetime
import subprocess

# Main idea: start N subprocesses, each with a different port. Have each julia process
# make a step and return a message after it receives an update.


def send(data, connection, msg_len=128):
    msg = bytes(" ".join(["{:.6e}".format(value) for value in data]) + "\n", 'UTF-8')
    padded_message = msg[:msg_len].ljust(msg_len, b'\0')
    #print("Sending:", padded_message)
    connection.sendall(padded_message)


def accept_connection(server_sock, ports):
    conn, addr = server_sock.accept()
    laddr = conn.getsockname()
    iWorker = ports.index(laddr[1])
    conn.setblocking(False)
    print(f"Connection from port {laddr}")
    sel.register(conn, selectors.EVENT_READ, handle_client)
    return iWorker, True, None


def handle_client(conn, ports):
    try:
        data = conn.recv(128)
        if data:
            laddr = conn.getsockname()
            raddr = conn.getpeername()
            iWorker = ports.index(laddr[1])

            print(f"Received from worker {iWorker:d} on {laddr}: {data.decode()}")
            
            send([10., 20., 30.], conn)
            
            vals = data.decode().split()
            vals = [int(vals[0])] + [float(v) for v in vals[1:]]
            return iWorker, True, vals
            
        else:
            laddr = conn.getsockname()
            iWorker = ports.index(laddr[1])
            print(f"Client {iWorker:d} on {laddr} disconnected")
            sel.unregister(conn)
            conn.close()
            return iWorker, False, None

    except ConnectionResetError:
        print("Connection reset by peer")
        sel.unregister(conn)
        conn.close()


def kill_process(simulation_process):
    if simulation_process.poll() is None:
        # Create a killfile.
#                with open(os.path.join(self.episode_dir, "killfile"), "w") as f:
#                    f.write("Terminate, terminate!")
            
        # Write some random data to the socket to prevent it locking up.
        # (Guess how I found out this was necessary?)
#                send(np.zeros(self.n_action_vars), self.connection, msg_len=self.msg_len)

        # Wait a bit to give Julia time to wrap things up in an orderly fashion.
#                time.sleep(self.kill_time_delay)
        
        # Force kill.
        simulation_process.terminate()
        simulation_process.wait()


# Create the sockets used for comms.
sel = selectors.DefaultSelector()
server_sockets = []
ports = [8089, 8089+1, 8089+2]

for port in ports:
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("localhost", port))
    server_sock.listen(5)
    server_sock.setblocking(False)
    sel.register(server_sock, selectors.EVENT_READ, accept_connection)
    server_sockets.append(server_sock)

print("Servers are listening on ports:", ports)

# Main loop to submit jobs, poll events, and get data back.
collectedData = []
isRunning = [False for _ in range(len(ports))]
simulation_processes = [None for _ in range(len(ports))]
jobIds = [-1 for _ in range(len(ports))]
jobCount = 0

while len(collectedData) < 50:
    time.sleep(0.5)
    
    # Check if all jobs slots are active. Submit a new simulation if not.
    for iWorker, port in enumerate(ports):
        if not isRunning[iWorker]:
            jobCount += 1
            
            jobIds[iWorker] = jobCount
            simulation_processes[iWorker] =  subprocess.Popen(
                ["julia",  "worker.jl", f"{port:d}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            isRunning[iWorker] = True
            
            print(f"Started sim {jobCount:d} using slot {iWorker:d} on port {port:d}")
            
            # Wait some time to artificially desynchronise the processes, emulating what
            # working with CFD might look like.
            time.sleep(0.5 + np.random.rand()*2)

    #print("I'm alive")
    events = sel.select(timeout=None)
    #print(events)
    for key, mask in events:
        callback = key.data
        iWorker, status, values = callback(key.fileobj, ports)
        print(f"  {iWorker:d} returned:", status, values)
        
        if status and values is not None:
            # Regular time step. Keep data.
            now = datetime.datetime.now()
            collectedData.append([jobIds[iWorker], iWorker, now] + values)
        
        elif status and values is None:
            # First handshake with no valid data. Ignore.
            pass

        else:
            # Job terminated and worker disconnected.
            isRunning[iWorker] = False
            
            # Ensure proper clean up. This shouldn't be necessary here as
            # simulation_process.poll() will not be None at this point, but you
            # never know.
            kill_process(simulation_processes[iWorker])
            print(f"Worker {iWorker:d} killed the subprocess properly")

# Terminate running jobs.
for iWorker in range(len(ports)):
    if isRunning[iWorker]:
        kill_process(simulation_processes[iWorker])
        print(f"Worker {iWorker:d} killed the subprocess properly")

# Get data into orderly format.
collectedData = pandas.DataFrame(data=np.array(collectedData),
    columns=["iJob", "iWorker", "time"] + [f"v{i:d}" for i in range(len(collectedData[0])-3)])

print(collectedData)

