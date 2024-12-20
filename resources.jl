using Plots
using Measures

# NOTE: stolen from WaterLily.jl/examples/TwoD_plots.jl
function flood(f::Array; shift=(0.,0.), cfill=:RdBu_11, clims=(), levels=10, kv...)
    if length(clims)==2
        @assert clims[1]<clims[2]
        @. f=min(clims[2], max(clims[1], f))
    else
        clims = (minimum(f), maximum(f))
    end
    Plots.contourf(axes(f,1).+shift[1],axes(f,2).+shift[2],f',
        linewidth=0, levels=levels, color=cfill, clims = clims, 
        aspect_ratio=:equal; kv...)
end

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

function plot_snapshot(sim, x0, R, x0Start, Rstart, x0End, Rend, rlDomainMin,
        rlDomainMax, episode_history, pos, vSet, vFlow, xStart, xEnd, t, figDir)
        
    # Plot the flow.
    p = flood(sim.flow.u[:, :, 1]; shift=(-0.5, -0.5), clims=(-0.5, 1.5), size=(1200, 800), margins=5mm,
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
    
    return fname
end

