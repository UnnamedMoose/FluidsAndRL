using Sockets
using Printf

function send(data, sock; length=128, counter=nothing)
    """ Send a series of floats. Optionally, prepend an int at the start of the message. """
    if counter == nothing
        prefix = ""
    else
        prefix = string(counter)
    end
    formatted_data = map(x -> @sprintf("%.6e", x), data)
    message = join(formatted_data, " ")
    padded_message = rpad(prefix * " " * message, length, " ")
    write(sock, padded_message)
    flush(sock)
end

function receive(sock; length=128, T=Float32)
    """ Receive a series of floats and parse them. """
    msg = read(sock, length)
    msg = strip(String(msg), '\0')
    data = [parse(T, value) for value in split(msg)]
    return data
end


socket_id = parse(Int, ARGS[1])

# Connect to Python server
sock = connect("127.0.0.1", socket_id)

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

