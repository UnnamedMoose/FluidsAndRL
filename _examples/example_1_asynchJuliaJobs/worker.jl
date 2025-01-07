using Sockets
using Printf

function send(data, sock; length=64, counter=nothing)
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

function receive(sock; length=64, T=Float32)
    """ Receive a series of floats and parse them. """
    msg = read(sock, length)
    msg = strip(String(msg), '\0')
    data = [parse(T, value) for value in split(msg)]
    return data
end


socket_id = parse(Int, ARGS[1])

# Connect to Python server
sock = connect("127.0.0.1", socket_id)

# Imitate a CFD time loop with an unknown number of time steps.
data = [1.0, 2.0, 3.0]
msg_len = 128

nSteps = rand(5:10)
for iStep in 1:nSteps
    # Send the data.
    send(data, sock; length=msg_len, counter=iStep)
    
    # Receive a vector of values.
    actions = receive(sock; length=msg_len, T=Float32)
    println(iStep, ": ", actions)
    
    sleep(1)
end

close(sock)

