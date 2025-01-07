import socket
import time
import selectors
#import os
#import numpy as np
#import pandas
#import datetime
#import torch
#import subprocess

# Main idea: start N subprocesses, each with a different port. Have each julia process
# make a step and return a message after it receives an update.

def send(data, connection, msg_len=128):
    msg = bytes(" ".join(["{:.6e}".format(value) for value in data]) + "\n", 'UTF-8')
    padded_message = msg[:msg_len].ljust(msg_len, b'\0')
    print("Sending:", padded_message)
    connection.sendall(padded_message)


def accept_connection(server_sock):
    conn, addr = server_sock.accept()
    conn.setblocking(False)
    print(f"Connection from {addr}")
    sel.register(conn, selectors.EVENT_READ, handle_client)


def handle_client(conn):
    try:
        data = conn.recv(128)
        if data:
            laddr = conn.getsockname()
            raddr = conn.getpeername()
        
            print(f"Received: {data.decode()} from {laddr}")
            
            send([10., 20., 30.], conn)
            #conn.sendall(f"Echo: {data.decode()}".encode())
        else:
            print("Client disconnected")
            sel.unregister(conn)
            conn.close()
    except ConnectionResetError:
        print("Connection reset by peer")
        sel.unregister(conn)
        conn.close()


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

# Main loop to poll events
while True:
    time.sleep(0.5)
    print("I'm alive")
    events = sel.select(timeout=None)
    print(events)
    for key, mask in events:
        callback = key.data
        callback(key.fileobj)

"""
while True:
    connection, address = server.accept()
    print("got con")
    buf = connection.recv(64).decode("utf-8")
    if len(buf) > 0:
        print("Python got:", buf)
        length = 64
        msg = bytes("got data! 1 2 3\n", 'UTF-8')
        padded_message = msg[:length].ljust(length, b'\0')
        # connection.send(len(msg))
        connection.sendall(padded_message)
    time.sleep(0.5)

connection.close()
"""

"""
nJobs = 1
baseSocketNo = 8089

sockets = []
connections = []
for iJob in range(nJobs):
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind(('localhost', baseSocketNo + iJob))
    serversocket.listen(5)
    sockets.append(serversocket)
    
    print(iJob, "is set up")
    time.sleep(1)
    print(iJob, "wants to accept")
    connection, address = serversocket.accept()
    connections.append(connection)
    
    print(iJob, "is listening on", baseSocketNo + iJob)

while True:
    time.sleep(0.5)

    for iJob in range(nJobs):
        buf = connections[iJob].recv(64).decode("utf-8")
        if len(buf) > 0:
            print(iJob, "got:", buf)
            length = 64
            msg = bytes(f"{iJob:d} got data! 1 2 3\n", 'UTF-8')
            padded_message = msg[:length].ljust(length, b'\0')
            connection.sendall(padded_message)

for connection in connections:
    connection.close()
"""

