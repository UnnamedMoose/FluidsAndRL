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
# For TCP message size. Needs to be consistent with what the server uses.
msg_len = 128
# Types matter!
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
# How close is close enough.
rThresholdEnd = 0.1

# Generate random start and end points.
xStart = random_point_in_circle(x0Start[1], x0Start[2], Rstart)
xEnd = random_point_in_circle(x0End[1], x0End[2], Rend)
pos = copy(xStart)

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
reward = 0
obs = nothing
done = 0
dTargetOld = nothing
iStep = 0
for t in timevals
    global iStep
    global pos
    global reward
    global obs
    global done
    global dTargetOld
    
    # Increment the time.
    iStep += 1
    println("Current time ", t, " time step ", iStep)

    # Advance the flow.    
    sim_step!(sim, t)

    # Accept action and update the state.
    if iStep > n_steps_startup
        
        # Retrieve the flow velocity at the agent location.
        vFlow = [flowInterp(pos, sim.flow.u[:, :, 1]),
                flowInterp(pos, sim.flow.u[:, :, 2])]

        # Compute the observation variables.
        obs, dTarget = assembleStateVector(xEnd, pos, vFlow)
        if dTargetOld == nothing
            dTargetOld = dTarget
        end
        
        # Send observation data. On the first pass this will be returned by the
        # reset() function on the python side with a dummy reward
        sendMessage(vcat(obs, reward, done), sock; length=msg_len, counter=iStep)
        
        # Receive the action vector (in this case one value).
        msg = read(sock, msg_len)
        msg = strip(String(msg), '\0')
        actions = [parse(T, action) for action in split(msg)]

        # Translate the action value in range <-1, 1> into set heading in range <0, 2pi>
        theta = max(0, min(2*pi, (actions[1]+1)*pi))

        # Set velocity along the desired heading.
        vSet = [cos(theta), sin(theta)] .* Vmax
        
        # Update episode stats.
        push!(episode_history, vcat(pos, vFlow, vSet, dTarget))

        # Update position using the Euler approach.
        pos = pos .+ (vSet .+ vFlow) .* dt
        println("  x_swimmer=", pos, " d_target=", dTarget, " action=", actions[1])
        
        # Compute the reward for the last action. This will be returned during the next pass.
        # The same as in Gunnarson et al. (2021), eq. 3
        rewward = -dt + 10*(dTargetOld - dTarget)/Vmax

        # Also check if outside of bounds.
        done = 0

        if (dTarget < rThresholdEnd)
            done = 1
            reward += 200
        end
        if (pos[1] < rlDomainMin[1]) || (pos[1] > rlDomainMax[1]) || (pos[2] < rlDomainMin[2]) || (pos[2] > rlDomainMax[2])
            done = 1
            reward -= 200
        end
                            
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

