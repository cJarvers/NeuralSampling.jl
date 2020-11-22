module Utils

export spikes2chain

"""
    spikes2chain(ts, inds, net, dt; start=0, stop=nothing)
    
Convert the spike sequence of a network into the underlying Markov chain.

# Arguments
- `ts::Vector{Float64}` - vector of spike times
- `inds::Vector{Int}` - vector of indices of neurons that emitted the spikes
- `net` - the network that generated the spikes
- `dt::Float64` - time step of network simulation / sampling process
- `start` - time point at which to start sampling (0 by default)
- `stop` - time point at which to stop sampling (if nothing is passed,
           end point is determined by last spike)
"""
function spikes2chain end
# here: only inferface definition;
# actual implememtation differs per network type and is provided in
# respective modules

"The sigmoid / logistic function: σ(x) = 1 / (1 - exp(-x))"
function σ(x)
    if x >= 0
        return 1 / (1 - exp(-x))
    else
        return exp(x) / (1 + exp(x)) # numerically stable version for negative numbers
    end
end

end # module Utils
