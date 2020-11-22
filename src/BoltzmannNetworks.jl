"""
    BoltzmannNetworks
   
Implements neural sampling in second-order Boltzmann distributions as presented in

> Lars Buesing, Johannes Bill, Bernhard Nessler and Wolfgang Maass (2011).
> Neural Dynamics as Sampling: A Model for Stochastic Computation in Recurrent Networks of Spiking Neurons.
> PLoS Computational Biology 7(11), e1002211.

The goal is to approximate a distribution ``p(\\mathbf{z})`` over the binary variables
``\\mathbf{z} = z_1, z_2, \\ldots, z_k`` by a neural network ``\\mathcal{N}`` of neurons
``v_1, v_2, \\ldots, v_k``. The spiking activity of ``\\mathcal{N}`` is interpreted as a
Markov chain implementing Markov Chain Monte Carlo. Specifically, a spike of neuron ``v_i``
sets the variable ``z_i`` to `true` for ``\\tau`` timesteps / milliseconds / samples. The
passing of this period is kept track of via auxiliary variables ``\\zeta_1, \\zeta_2, \\ldots, \\zeta_k``.

Networks are represented by the type `BoltzmannNetwork`. This module implements the
following methods from the paper:

- sampling with absolute refractory period in discrete time via the function `sample_absolute!`
- sampling with relative refractory period in discrete time via the function `sample_relative!`
- Gibbs sampling (not using neural dynamics) as a baseline method

"""
module BoltzmannNetworks

using ..Utils: σ
import ..Utils: spikes2chain
import Base: show, length

export BoltzmannNetwork
export randomnetwork, sample_absolute!, sample_relative!, sample_clamp!

###############################################################################
# Network type                                                                #
###############################################################################
"""
    BoltzmannNetwork

Represents a network of spiking neurons that performs MCMC sampling from the distribution

```math
p(\\mathbf{z}) = \\frac{1}{Z} \\exp \\left( \\sum_{i,j} \\frac{1}{2} W_{ij} z_i z_j + \\sum_i b_i z_i \\right)
```

where `Z` is the partition function, `W` are weights and `b` are biases.

# Fields
- `ζ::Vector{Int}` - vector of auxiliary variables `ζ`
- `z::Vector{Bool}` - vector of variables `z`
- `u::Vector{Float64}` - vector of neural membrane potentials
- `W::Matrix{Float64}` - weights relating the variables (symmetric, zeros on main diagonal)
- `b::Vector{Float64}` - biases governing neural excitability
- `τ::Int` - time constant encoding for how many steps `z` is `true` after a spike
- `g::G` - refractoriness function for sampling with relative refr. period
- `f::F` - activation function for sampling with relative refr. period
"""
struct BoltzmannNetwork{G, F}
    ζ::Vector{Int}
    z::Vector{Bool}
    u::Vector{Float64}
    W::Matrix{Float64}
    b::Vector{Float64}
    τ::Int
    g::G
    f::F
    
    # constructor to enforce constraints on W
    function BoltzmannNetwork(ζ, z, u, W, b, τ, g, f)
        for i in 1:size(W, 1)
            @assert W[i, i] == 0.0 "Main diagonal of weight matrix has to be all zeros."
            for j in i:size(W, 2)
                @assert W[i, j] == W[j, i] "Weight matrix has to be symmetric."
            end
        end
        return new{typeof(g), typeof(f)}(ζ, z, u, W, b, τ, g, f)
    end
end

"Generate a random `BoltzmannNetwork` of `n` neurons."
function randomnetwork(n::Int; τ=20, g=g) # τ as in Buesing et al. (2011)
    ζ = zeros(Int, n)
    z = zeros(Bool, n)
    u = zeros(Float64, n)
    W = randn(n, n) * 0.3 # scaling as in Buesing et al. (2011), Fig. 3, 5, 6, 7
    for i in 1:n, j in i:n
        if i==j
            W[i, j] = 0. # enforce that main axis is all zeros
        else
            W[i, j] = W[j, i] # enforce that weights are symmetric
        end
    end
    b = randn(n) * 0.5 .- 1.5 # scaling as in Buesing et al. (2011), Fig. 3, 5, 6, 7
    
    f = gettransferfun(g, τ=τ)
    return BoltzmannNetwork(ζ, z, u, W, b, τ, g, f)
end


###############################################################################
# Sampling with absolute refractory period.                                   #
###############################################################################
"""
    step_absolute!(net::BoltzmannNetwork)

Implements a single step of neural MCMC sampling with absolute refractory period
in discrete time.

# Details

Alters the state of the membrane potentials `net.u`, the variables `net.z` and
the auxiliary variables `net.ζ`. The neurons are updated sequentially (as
explained in the paper, this is necessary for correct sampling). For each
neuron, the membrane potential is calculated in accordance with the
**neural computability condition**:

```math
u_i = b_i + sum_j W_ij * z_i * z_j
```

If the neuron is refractory (``\\zeta \\ge 1``), ζ is decreased deterministically.
Otherwise the neurons spikes with probability ``\\sigma(u - \\log(\\tau))``. A spike
sets `z` to `true` and `ζ` to `τ`.
"""
function step_absolute!(net::BoltzmannNetwork)
    spikes = zeros(Bool, length(net.z))
    # update neurons
    # (in order, since transition operators have to be applied sequentially)
    for i in 1:length(spikes)
        # update membrane potential
        net.u[i] = net.b[i] + net.W[i, :]' * net.z
        # generate spike
        if (net.ζ[i] ≤ 1) && (rand() < σ(net.u[i] - log(net.τ)))
            spikes[i] = true
            net.ζ[i] = net.τ
            net.z[i] = 1
        else
            net.ζ[i] = max(net.ζ[i] - 1)
            if net.ζ[i] == 0
                net.z[i] = 0
            end
        end
    end
    return(spikes)
end

"""
    sample_absolute!(net, steps, burnin, dt)
    
Performs neural sampling with absolute refractory period in discrete time in
`BoltzmannNet` `net` for `steps` steps after a burnin time of `burnin` steps
with step size `dt`.

# Returns
- `ts::Vector{Float64}` - vector of spike times
- `inds::Vector{Int}` - vector of neuron indices of neurons that spiked 
"""
function sample_absolute!(net, steps, burnin, dt)
    for _ in 1:burnin
        step_absolute!(net)
    end
    ts = zeros(0)
    inds = zeros(Int, 0)
    for t in 1:steps
        spikes = step_absolute!(net)
        for spike in findall(spikes)
            push!(ts, t * dt)
            push!(inds, spike)
        end
    end
    return ts, inds
end

###############################################################################
# Sampling with relative refractory period                                    #
###############################################################################
#
# In sampling with a relative refractory period, the neural transfer function
# is slightly more complicated. The refactoriness of a neuron is given by the
# function g(ζ): [0, τ] -> [0, 1]
#
# Depending on the exact function g chosen, the neural transfer function f
# has to fulfill some specific conditions. In general, a suitable f exists but
# cannot be found analytically. Therefore, it has to be approximated numerically.
#
# First, we implement the function g given in the paper.
"""Refractory function g(ζ).
Quantifies how excitable a neuron is based on variable ζ,
which encodes the time since last spike.

Assumes that ζ is in range [0, 1]"""
function g(ζ)
    x = 1 - ζ + sin(2π * ζ) / (2π) # according to Table 1 in Buesing et al. (2011)
    return min(1, max(0, x))
end

"Alternative refractory function"
function g2(ζ) # assumes that ζ is in range [0, 1]
    x = 4 * (1 - ζ) + sin(8π * ζ) / (2π) # according to Table 1 in Buesing et al. (2011)
    return min(1, max(0, x))
end

"Alternative refractory function"
function g3(ζ) # assumes that ζ is in range [0, 1]
    x = 1 - 2ζ + sin(4π * ζ) / (2π) # according to Table 1 in Buesing et al. (2011)
    return min(1, max(0, x))
end

# Next, we need to approximate f for a given g.
"""
    gettransferfun(g; τ=20)

Approximates the neural transfer function `f(u)` for a given function `g(ζ)`.

# Arguments
- `g::Function` - refractoriness; should accept values in range [0, 1].
"""
function gettransferfun(g; τ=20)
    # the following function calculates the error for a given u and f(u)
    ferror(u, fu) = abs(exp(u) - fu * 
        sum(prod(1 - g(ζ/τ) * fu for ζ in η+1:τ) for η in 1:τ-1) /
        prod(1 - g(ζ/τ) * fu for ζ in 1:τ))
    # we assume that membrane potentials will stay in the range [-10, 10]
    us = -10:0.01:10
    fs = zeros(length(us))
    # we know that the output of f must lie in [0, 1]
    vals = 0:0.001:1
    # now, for each possible input value u, find the output value that minimizes the error
    for (i, u) in enumerate(us)
        fs[i] = vals[argmin(ferror.(u, vals))]
    end
    # now we can generate the function f:
    # for each input value, find the nearest known input value from us and
    # return the corresponding f(u)
    function f(u; us=us, fs=fs)
        i = argmin(abs.(u .- us))
        return fs[i]
    end
    return f
end

# Given the correct transfer function, we can implement the sampling
"""
    step_relative!(net::BoltzmannNetwork)

Implements a single step of neural MCMC sampling with relative refractory period
in discrete time.

# Details

Alters the state of the membrane potentials `net.u`, the variables `net.z` and
the auxiliary variables `net.ζ`. The neurons are updated sequentially (as
explained in the paper, this is necessary for correct sampling). For each
neuron, the membrane potential is calculated in accordance with the
**neural computability condition**:

```math
u_i = b_i + sum_j W_ij * z_i * z_j
```

The neural refactoriness is given by `g(ζ)`. At each point, the neuron fires with
probability `g(ζ) * f(u)`. A spike sets `z` to `true` and `ζ` to `τ`. If the neuron
does not spike, `ζ` decreases and `z` is set to `false` once `ζ` becomes `0`.
"""
function step_relative!(net::BoltzmannNetwork)
    spikes = zeros(Bool, length(net.z))
    # update neurons
    # (in order, since transition operators have to be applied sequentially)
    for i in 1:length(spikes)
        # update membrane potential
        net.u[i] = net.b[i] + net.W[i, :]' * net.z
        # generate spike with probability g(ζ) * f(u), otherwise decay ζ
        if (rand() < net.g(net.ζ[i] / net.τ) * net.f(net.u[i]))
            spikes[i] = true
            net.ζ[i] = net.τ
            net.z[i] = 1
        else
            net.ζ[i] = max(net.ζ[i] - 1)
            if net.ζ[i] == 0
                net.z[i] = 0
            end
        end
    end
    return(spikes)
end

"""
    sample_relative!(net, steps, burnin, dt)
    
Performs neural sampling with relative refractory period in discrete time in
`BoltzmannNet` `net` for `steps` steps after a burnin time of `burnin` steps
with step size `dt`.

# Returns
- `ts::Vector{Float64}` - vector of spike times
- `inds::Vector{Int}` - vector of neuron indices of neurons that spiked 
"""
function sample_relative!(net, steps, burnin, dt)
    # run the Markov chain for burn-in period
    for _ in 1:burnin
        step_relative!(net)
    end
    
    # sample spikes 
    ts = zeros(0)
    inds = zeros(Int, 0)
    for t in 1:steps
        spikes = step_relative!(net)
        for spike in findall(spikes)
            push!(ts, t * dt)
            push!(inds, spike)
        end
    end
    return ts, inds
end

###############################################################################
# Sampling with clamped values (inferring posterior)                          #
###############################################################################
"""
    step_clamp!(net::BoltzmannNetwork, clamp_on, clamp_off)

Performs one step of neural sampling with relative refractory period in `net`
while clamping the neurons at indices `clamp_on` to `true` and the neurons at
indices `clamp_off` to `false`.
"""
function step_clamp!(net::BoltzmannNetwork, clamp_on, clamp_off)
    spikes = zeros(Bool, length(net.z))
    # update neurons
    # (in order, since transition operators have to be applied sequentially)
    for i in shuffle(1:length(spikes))
        if i in clamp_on
            net.z[i] = true
            continue
        end
        if i in clamp_off
            net.z[i] = false
            continue
        end
        # update membrane potential
        net.u[i] = net.b[i] + net.W[i, :]' * net.z
        # generate spike with probability g(ζ) * f(u), otherwise decay ζ
        if (rand() < net.g(net.ζ[i] / net.τ) * net.f(net.u[i]))
            spikes[i] = true
            net.ζ[i] = net.τ
            net.z[i] = 1
        else
            net.ζ[i] = max(net.ζ[i] - 1)
            if net.ζ[i] == 0
                net.z[i] = 0
            end
        end
    end
    return(spikes)
end

"""
    sample_clamp!(net, steps, burnin, dt, clamp_on, clamp_off)
    
Performs neural sampling with relative refractory period in discrete time in
`BoltzmannNet` `net` for `steps` steps after a burnin time of `burnin` steps
with step size `dt` while clamping the neurons at indices `clamp_on` to `true`
and the neurons at indices `clamp_off` to `false`.

# Returns
- `ts::Vector{Float64}` - vector of spike times
- `inds::Vector{Int}` - vector of neuron indices of neurons that spiked 
"""
function sample_clamp!(net, steps, burnin, dt, clamp_on, clamp_off)
    # run the Markov chain for burn-in period
    for _ in 1:burnin
        step_clamp!(net, clamp_on, clamp_off)
    end
    
    # sample spikes 
    ts = zeros(0)
    inds = zeros(Int, 0)
    for t in 1:steps
        spikes = step_clamp!(net, clamp_on, clamp_off)
        for spike in findall(spikes)
            push!(ts, t * dt)
            push!(inds, spike)
        end
    end
    return ts, inds
end

###############################################################################
# Conversion of spike sequence into Markov chain                              #
###############################################################################
function spikes2chain(ts, inds, net, dt; start=0, stop=nothing)
    if stop == nothing
        stop = maximum(ts) + net.τ * dt
    end
    timesteps = round(Int, (stop - start) / dt)
    chain = zeros(Bool, length(net), timesteps)
    for (t, i) in zip(ts, inds)
        # each spike sets the corresponding variable to true for τ timesteps
        tᵢₙₜ = max(1, round(Int, (t - start) / dt))
        chain[i, tᵢₙₜ:min(timesteps, tᵢₙₜ+net.τ)] .= true
    end
    return chain
end

###############################################################################
# Overloads for pretty printing etc.                                          #
###############################################################################
function Base.show(io::IO, net::BoltzmannNetwork)
    print(io, "BoltzmannNetwork of $(length(net.z)) neurons.")
end

function Base.show(io::IO, ::MIME"text/plain", net::BoltzmannNetwork)
    print(io,
        "BoltzmannNetwork of $(length(net.z)) neurons.\n",
        "\t z = $(net.z) \n",
        "\t u = $(net.u) \n",
        "\t W = $(net.W) \n",
        "\t b = $(net.b) \n",
        "\t τ = $(net.τ)"
    )
end

function Base.length(net::BoltzmannNetwork)
    return length(net.z)
end

end # module BoltzmannNetworks
