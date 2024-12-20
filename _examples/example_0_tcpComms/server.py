# v0

import socket
import time

serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind(('localhost', 8089))
serversocket.listen(5) # become a server socket, maximum 5 connections
print("Listening")

while True:
    connection, address = serversocket.accept()
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
import socket

def send_message(conn, message, length=32):
    # Pad or truncate the message to the fixed length
    encoded_message = message.encode("utf-8")
    padded_message = encoded_message[:length].ljust(length, b'\0')  # Null padding
    conn.sendall(padded_message)

def receive_message(conn, length=32):
    # Read exactly 'length' bytes
    message_data = conn.recv(length)
    return message_data.decode("utf-8").rstrip('\0')  # Remove padding

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("127.0.0.1", 5000))
server.listen(5)

print("Server listening on port 5000...")
conn, addr = server.accept()
print("Connected by", addr)

# # Send and receive fixed-length messages
# send_message(conn, "Hello from Python!")
# received = receive_message(conn)
# print("Received:", received)

conn.close()
"""
