
# v0

using Sockets
using Printf

# Connect to Python server
sock = connect("127.0.0.1", 8089)

# Send data
#write(sock, "hello.jl")

data = [1.0, 2.0, 3.0]
length = 64
formatted_data = map(x -> @sprintf("%.2e", x), data)
message = join(formatted_data, " ")
padded_message = rpad(message, length, " ")#[1:length]
write(sock, padded_message)
flush(sock)  # Ensure data is sent immediately

# Receive response
msg = read(sock, 64)
msg = strip(String(msg), '\0')
println(msg)

#response = read(sock, String)#, sizeof(Float32) * length(data))
#result = reinterpret(Float32, response)
#println("Response from Python:", response)

close(sock)
#=

using Sockets

function send_message(sock, message, length=32)
    # Pad or truncate the message to the fixed length
    padded_message = rpad(message, length, '\0')[1:length]
    write(sock, padded_message)
    flush(sock)  # Ensure data is sent immediately
end

function receive_message(sock, length=32)
    # Read exactly 'length' bytes
    data = read(sock, length)
    return strip(String(data), '\0')  # Remove padding
end

# Connect to the Python server
sock = connect("127.0.0.1", 5000)

# Send and receive fixed-length messages
send_message(sock, "Hello from Julia!")
received_message = receive_message(sock)
println("Received: ", received_message)

# Keep the connection open for further communication if needed
send_message(sock, "Another fixed-length message")

# Close the socket when done
close(sock)

=#
