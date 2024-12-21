using WaterLily
using CUDA
using StaticArrays
import Printf.@sprintf
using Plots
using LinearAlgebra: norm2
using DelimitedFiles
import Plots:Animation, buildanimation 
using Sockets
using Printf

include("resources.jl")

# Grab socket ID and episode ID from the args
episode_path = "episode_" * ARGS[1]
socket_id = parse(Int, ARGS[2])

# Connect to Python server
sock = connect("127.0.0.1", socket_id)

# Main settings.
T = Float32
# Reynolds number of the cylinder.
Re = T(250)
# Grid size.
N = 3*2^7
M = 2^8
# Circle radius.
RbyM = T(1/32)
# When set to below 1, this makes the problem impossible without RL and energy scavenging
Vmax = T(1.2)
dTargetThresholdSuccess = T(0.1)
dt = T(0.5)
# This excludes the initial ramp-up time used to let the flow develop
max_steps = 10
n_steps_startup = 2
# Circle radius and origin.
R = M*RbyM
x0 = T.([M/2, M/2])

# Assign values to the RL domain. Watch the types!
x0Start = T.([10., -4.1] .* R .+ x0)
Rstart = T(4*R)
x0End = T.([10., 4.1] .* R .+ x0)
Rend = T(4*R)
rlDomainMin = T.([1.0, -10.0] .* R .+ x0)
rlDomainMax = T.([20.0, 10.0] .* R .+ x0)

# Generate random start and end points.
xStart = random_point_in_circle(x0Start[1], x0Start[2], Rstart)
xEnd = random_point_in_circle(x0End[1], x0End[2], Rend)
pos = copy(xStart)

# Send the initial conditions to Python.
println("Julia sending initial")
sendMessage(vcat([xStart..., xEnd...]), sock)
println("Julia sent initial")

# === main ===
#@assert CUDA.functional()

if isdir(episode_path)
    println("Overwriting previous episode data! You have only yourself to blame for this!")
    rm(episode_path, force=true, recursive=true)
end
mkdir(episode_path)

U = 1
body = AutoBody((x,t)->√sum(abs2, x .- x0) - R)
sim = Simulation((N, M), (U, 0), R; ν=U*R/Re, body=body, T=T)#, mem=CuArray)

timevals = range(0, (n_steps_startup+max_steps)*dt, step=dt)
frames = []
episode_history = []

iStep = 0
for t in timevals
    global iStep
    global pos
    
    # Increment the time.
    iStep += 1
    println("Current time ", t, " time step ", iStep)

    # Advance the flow.    
    sim_step!(sim, t)

    #=
    data = [1.0, 2.0, 3.0]
    length = 128
    formatted_data = map(x -> @sprintf("%.2e", x), data)
    message = join(formatted_data, " ")
    padded_message = rpad(message, length, " ")
    println("Julia sending: ", padded_message)
    write(sock, padded_message)
    println("Julia sent")
    flush(sock)
    println("Julia flushed")
    
    # Receive response
    msg = read(sock, 128)
    msg = strip(String(msg), '\0')
    println("Julia got: ", msg)
    =#
    
    if iStep > n_steps_startup
        # Retrieve the flow velocity at the agent location.
        vFlow = [flowInterp(pos, sim.flow.u[:, :, 1]),
                flowInterp(pos, sim.flow.u[:, :, 2])]

       # Set the state and target distance.
    #state, dTargetOld = assembleStateVector(xEnd, pos, vFlow)
    
    # Translate the action value in range <-1, 1> into set heading in range <0, 2pi>
#    theta = max(0, min(2*pi, (a[1]+1)*pi))

    # Set velocity along the desired heading.
#    vSet = [cos(theta), sin(theta)] .* env.params.Vmax


        # Set velocity  to point directly towards the target.
        dTarget = norm2(xEnd .- pos)
        vSet = (xEnd .- pos) / dTarget * Vmax
           
        # Update position using the Euler approach.
        pos = pos .+ (vSet .+ vFlow) .* dt
        println("  x_swimmer=", pos, " d_target=", dTarget)
        
        # Send data
        sendMessage(vcat([pos..., vSet..., vFlow...]), sock; counter=iStep)
        
        # Receive response
        msg = read(sock, 128)
        msg = strip(String(msg), '\0')
        println("Julia got: ", msg)
        
        # Update episode stats.
        push!(episode_history, vcat(pos, vFlow, vSet, dTarget))
        
        # Plot the current state and save the fig into a file.        
        fname = plot_snapshot(sim, x0, R, x0Start, Rstart, x0End, Rend, rlDomainMin,
            rlDomainMax, episode_history, pos, vSet, vFlow, xStart, xEnd, iStep-n_steps_startup-1, episode_path)
        push!(frames, fname)
    end
end

# Convert frames to a gif.
anim = Animation(episode_path, frames)
buildanimation(anim, joinpath(episode_path, "animatedEpisode.gif"), fps=12, show_msg=false)

# Save all data to a csv.
open(joinpath(episode_path, "episodeHistory.csv"); write=true) do f
    write(f, "x, y, uFlow, vFlow, u, v, dTarget\n")
    writedlm(f, episode_history, ',')
end

