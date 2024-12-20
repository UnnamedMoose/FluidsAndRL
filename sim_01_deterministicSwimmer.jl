using WaterLily
using CUDA
using StaticArrays
#using GLMakie
import Printf.@sprintf
using Plots
using LinearAlgebra: norm2
using DelimitedFiles
#import Plots:Animation, buildanimation  

include("/home/artur/git/WaterLily.jl/examples/TwoD_plots.jl")

function flowInterp(xi, arr)
    # Index below the interpolation coordinate.
    ii = [
        max(1, min(size(arr, 1)-1, Int(floor(xi[1])))),
        max(1, min(size(arr, 2)-1, Int(floor(xi[2]))))]
    
    # Local coordinate system <0, 1>. delta is 1 in both directions
    a = 1. / ((2 - 1) * (2 - 1))
    xx = [ii[1]+1 - xi[1], xi[1] - ii[1]]
    yy = [ii[2]+1 - xi[2], xi[2] - ii[2]]
    
    # Interp
    data = arr[ii[1]:ii[1]+1, ii[2]:ii[2]+1]
    b = data*yy
    res = a*(transpose(xx)*b)
    return res
end

function random_point_in_circle(x0, y0, r)
    # Generate a random angle between 0 and 2π
    angle = rand() * 2*pi
    
    # Generate a random radius between 0 and r
    # Using sqrt(rand()) generates a radius that is proportional
    # to the square root of a uniformly distributed random variable,
    # which results in a uniform distribution of points within the circle.
    # This is known as the polar coordinates method for generating
    # random points within a circle.
    radius = sqrt(rand()) * r
    
    # Calculate the x and y coordinates of the point
    x_point = x0 + radius * cos(angle)
    y_point = y0 + radius * sin(angle)
    
    # Return the x and y coordinates as a tuple
    return vcat(x_point, y_point)
end

function circle_shape(x0, y0, r, nPts::Int=101)
    angles = LinRange(0, 2*pi, nPts)
    x = x0 .+ r .* cos.(angles)
    y = y0 .+ r .* sin.(angles)
    return x, y
end

function circle(n, m, R, center; Re=250, T=Float32, mem=Array)
    U = 1.0
    n = 3*2^7
    m = 2^8
    R = 32
    #ν = U*R/Re
    body = AutoBody((x,t)->norm2(x .- center) - R)
    #Simulation((n+2, m+2), [U, 0.], R; ν, body, T=T, mem=mem)
    return Simulation((n, m), (U, 0), R; ν=U*R/Re, body=body, T=T, mem=mem)
end

function assembleStateVector(xEnd, pos, vFlow)
    # State is relative position to the target (unit vector) and 
    # flow velocity at the swimmer's position (bounded to <-Uinf, Uinf>).
    vecToTarget = xEnd .- pos
    dToTarget = norm2(vecToTarget)
    vecToTarget /= max(1e-6, dToTarget)
    state = [
        vecToTarget[1],
        vecToTarget[2],
        max(-1, min(1, vFlow[1])),
        max(-1, min(1, vFlow[2]))
    ]
    return state, dToTarget
end

T = Float32
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
max_steps = 50
n_steps_startup = 100
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

# === main ===
#@assert CUDA.functional()

Re = 250
U = 1
body = AutoBody((x,t)->√sum(abs2, x .- x0) - R)
sim = Simulation((N, M), (U, 0), R; ν=U*R/Re, body=body, T=T)#, mem=CuArray)

timevals = range(0, (n_steps_startup+max_steps)*dt, step=dt)
frames = []
episode_history = []
figDir = "."

iStep = 0
for t in timevals
    global iStep
    global pos
    iStep += 1
    println("Current time ", t, " time step ", iStep)
    
    sim_step!(sim, t)
    
    # Retrieve the flow velocity at the agent location.
    vFlow = [flowInterp(pos, sim.flow.u[:, :, 1]),
            flowInterp(pos, sim.flow.u[:, :, 2])]
    
    # Set the state and target distance.
    #state, dTargetOld = assembleStateVector(xEnd, pos, vFlow)
    
    # Translate the action value in range <-1, 1> into set heading in range <0, 2pi>
#    theta = max(0, min(2*pi, (a[1]+1)*pi))

    # Set velocity along the desired heading.
#    vSet = [cos(theta), sin(theta)] .* env.params.Vmax
    if iStep > n_steps_startup
        dTarget = norm2(xEnd .- pos)
        vSet = (xEnd .- pos) / dTarget * Vmax
           
        # Update position using the Euler approach.
        pos = pos .+ (vSet .+ vFlow) .* dt
        println("x_swimmer=", pos, " d_target=", dTarget)
        
        # Update episode stats.
        push!(episode_history, vcat(pos, vFlow, vSet, dTarget))
        
        # Plot the flow.
        p = flood(sim.flow.u[:, :, 1]; shift=(-0.5, -0.5), clims=(-0.5, 1.5),
            xlabel="x [grid units]", ylabel="y [grid units]", colorbar_title="\$U_{x}/U_{\\infty}\$")

        # Plot the cylinder
        plot!(circle_shape(x0[1], x0[2], R), fillalpha=1,
            linecolor=:grey, c=:grey, lw=1, seriestype=[:shape,], label="")

        # Plot the start and end regions
        plot!(circle_shape(x0Start[1], x0Start[2], Rstart), fillalpha=0.1,
            linecolor=:green, c=:green, line=(1, :dash, 0.6), seriestype=[:shape,], label="")
        plot!(circle_shape(x0End[1], x0End[2], Rend), fillalpha=0.1,
            linecolor=:red, c=:red, line=(1, :dash, 0.6), seriestype=[:shape,], label="")

        # Plot the bounds for the agent.
        plot!(Shape([rlDomainMin[1], rlDomainMax[1], rlDomainMax[1], rlDomainMin[1]],
                    [rlDomainMin[2], rlDomainMin[2], rlDomainMax[2], rlDomainMax[2]]),
                    fillalpha=0.5, fillopacity=0.25, color=:grey, label="", lc=:black, line=(1, :dash, 0.6))

        # Plot the trajectory so far.
        plot!([x[1] for x in episode_history], [x[2] for x in episode_history], c=:purple, lw=2, label="Trajectory")

        # Plot the agent speed and flow velocity.
        vscale = 10
        plot!([pos[1], pos[1]+vSet[1]*vscale], [pos[2], pos[2]+vSet[2]*vscale], label="\$V_{agent}\$", lc=:black, lw=2)
        plot!([pos[1], pos[1]+vFlow[1]*vscale], [pos[2], pos[2]+vFlow[2]*vscale], label="\$V_{flow}\$", lc=:orange, lw=2)

        # Plot the start and end positions.
        scatter!([xStart[1]], [xStart[2]], label="Start", c=:green, ms=4, markerstrokewidth=0)
        scatter!([xEnd[1]], [xEnd[2]], label="End", c=:red, ms=4, markerstrokewidth=0)

        # Move the legend outside.
        plot!(legend=:outertop, legendcolumns=3)

        # Keep this plot for making an animation.
        fname = @sprintf("%06d.png", t)
        savefig(joinpath(figDir, fname))
        push!(frames, fname)
    end  # Swimmer is active.
end

open("episodeHistory.csv"; write=true) do f
    write(f, "x, y, uFlow, vFlow, u, v, dTarget\n")
    writedlm(f, episode_history, ',')
end

# TODO
# Save frames to a gif.
#anim = Animation(figDir, frames)
#buildanimation(anim, joinpath(figDir, "animatedEpisode.gif"), fps=1, show_msg=true)

