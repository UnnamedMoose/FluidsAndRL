# TODO
# - Create a subprocess that runs a single simulation with a given episode info
# - get the initial random conditions via tcp from WL
# - get state info after each time step via tcp from WL
# TODO

import subprocess
import socket
import time
import os

episodeId = "commsTestEpisode"

# Spawn the simulation process.
portNumber = 8091
simulation_process = subprocess.Popen(
    ["julia", "sim_02_swimmerClient.jl", episodeId, "{:d}".format(portNumber)],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    preexec_fn=os.setsid
)

# Wait a bit and open a socket.
time.sleep(1.)
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind(('localhost', portNumber))
serversocket.listen(5) # become a server socket, maximum 5 connections
connection, address = serversocket.accept()
print("Server listening")

try:
    # Receive the initial handshake with start conditions etc.
    while True:
        time.sleep(0.5)
        buf = connection.recv(128).decode("utf-8")
        if len(buf) > 0:
            print("Python got initial:", buf)
            break

    # Keep running the simulation.
    while True:
        # Chill for a bit.
        time.sleep(0.5)
        print("I'm alive")
        
        # Check if the process has finished
        return_code = simulation_process.poll()
        if return_code is not None:
            print(f"Simulation finished with exit code {return_code}")
            
            # Capture all the outputs when the process finishes
            stdout_data, stderr_data = simulation_process.communicate()
            print("Final output:")
            print(stdout_data.decode())
            print(stderr_data.decode())

            # Close the socket.
            connection.close()
        
            break
                
        # Send a message and wait for return.
        buf = connection.recv(128).decode("utf-8")
        if len(buf) > 0:
            print("Python got:", buf)
            length = 128
            msg = bytes("got data! 1 2 3\n", 'UTF-8')
            padded_message = msg[:length].ljust(length, b'\0')
            print("Python sending:", padded_message)
            connection.sendall(padded_message)
       
except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Cleanup: Close TCP connection and terminate the subprocess if still running
    print("Cleaning up...")
    
    if simulation_process.poll() is None:
        print("Terminating simulation process...")
        simulation_process.terminate()  # Or simulation_process.kill()
        simulation_process.wait()  # Wait for it to terminate
    
"""
stdout_data, stderr_data = simulation_process.communicate()
print(stdout_data.decode())
"""
