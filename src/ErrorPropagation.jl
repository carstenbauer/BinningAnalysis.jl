using ForwardDiff, Statistics


################################################################################
### Common code
################################################################################

abstract type Abstract_Error_Propagator end


function var_O1(g::Function, ep::Abstract_Error_Propagator)
    # var / N because we actually want Var(x_mean), not Var(x)
    x_mean = xmean(ep)
    x_varN = xvar(ep) / ep.N
    y_mean = ymean(ep)
    y_varN = yvar(ep) / ep.N
    xy_covN = xycov(ep) / ep.N

    dgx = ForwardDiff.derivative(x -> g(x, y_mean), x_mean)
    dgy = ForwardDiff.derivative(y -> g(x_mean, y), y_mean)

    # from Quantum Monte Carlo Methods p56
    dgx^2 * x_varN + dgy^2 * y_varN + 2.0 * dgx * dgy * xy_covN
end

Statistics.std(g::Function, ep::Abstract_Error_Propagator) = sqrt(var_O1(g, ep))


################################################################################
### Version 1: Using direct calculation of variance and covariance
################################################################################


mutable struct Error_Propagator1{T} <: Abstract_Error_Propagator
    # ∑x
    sums1D::Vector{T}
    #∑xy
    sums2D::Matrix{T}

    N::Int64
end

# Initialization
function Error_Propagator1(zero::T, N::Integer) where T
    Error_Propagator1(
        [copy(zero) for _ in 1:N],
        [copy(zero) for _ in 1:N, __ in 1:N],
        0
    )
end

function Base.push!(ep::Error_Propagator1{T}, args::T...) where T
    ep.N += 1
    @simd for i in eachindex(ep.sums1D)
        ep.sums1D[i] += args[i]
        for j in eachindex(ep.sums1D)
            ep.sums2D[i, j] += args[i] * args[j]
        end
    end
    nothing
end


means(ep::Error_Propagator1) = ep.sums1D ./ ep.N
function covmat(ep::Error_Propagator1)
    invN = 1.0 / ep.N
    invN1 = 1.0 / (ep.N-1)
    [
        (ep.sums2D[i, j] - ep.sums1D[i] * ep.sums1D[j] * invN) * invN1
        for i in eachindex(ep.sums1D), j in eachindex(ep.sums1D)
    ]
end

function var_O1(g::Function, ep::Error_Propagator1)
    # NOTE: Not type stable :(
    # derivatives = ForwardDiff.gradient(v -> g(v...), means(ep))
    # NOTE: Also not type stable
    # NOTE: crossing a function barrier doesn"t hel either
    ms = means(ep)
    derivatives = [
        ForwardDiff.derivative(
            x -> g(ms[1:i-1]..., x, ms[i+1:end]...),
            ms[i]
        ) for i in eachindex(ms)
    ]


    result = 0.0
    invN = 1.0 / ep.N
    invNN1 = 1.0 / (ep.N * (ep.N-1))
    for i in eachindex(ep.sums1D)
        for j in eachindex(ep.sums1D)
            result += derivatives[i] * derivatives[j] * (
                ep.sums2D[i, j] - ep.sums1D[i] * ep.sums1D[j] * invN
            ) * invNN1
        end
    end

    result
end


################################################################################
### Version 2: Using Welford's Algorithm for the variance and covariance
################################################################################


mutable struct Error_Propagator2 <: Abstract_Error_Propagator
    x_mean::Float64
    y_mean::Float64

    x_var::Float64
    y_var::Float64
    xy_cov::Float64
    N::Int64
end

# Initialization
Error_Propagator2() = Error_Propagator2(0., 0., 0., 0., 0., 0)

function Base.push!(ep::Error_Propagator2, x::Float64, y::Float64)
    # Common Welford stuff
    ep.N += 1
    dx = x - ep.x_mean
    ep.x_mean += dx / ep.N
    dy = y - ep.y_mean
    ep.y_mean += dy / ep.N

    # The rest can be skipped on the first run.
    # ep.N < 2 && return nothing
    # vars
    dx2 = x - ep.x_mean
    ep.x_var += dx * dx2
    dy2 = y - ep.y_mean
    ep.y_var += dy * dy2

    # covar
    ep.xy_cov += dx * dy2
    nothing
end

# These require a final normalization
# (mean's do not though)
xmean(ep::Error_Propagator2) = ep.x_mean
ymean(ep::Error_Propagator2) = ep.y_mean
xvar(ep::Error_Propagator2) = ep.x_var / (ep.N - 1)
yvar(ep::Error_Propagator2) = ep.y_var / (ep.N - 1)
xycov(ep::Error_Propagator2) = ep.xy_cov / (ep.N - 1)


################################################################################
### Version 3: Add a Binning Analysis to Version 1
################################################################################

mutable struct EPCompressor{T}
    values::Vector{T}
    switch::Bool
end

struct Error_Propagator3{T, N} <: Abstract_Error_Propagator
    compressors::NTuple{N, EPCompressor{T}}
    # ∑x
    sums1D::NTuple{N, Vector{T}}
    #∑xy
    sums2D::NTuple{N, Matrix{T}}

    count::Vector{Int64}
end

# Initialization
function Error_Propagator3(zero::T, N_arguments::Integer, N_levels=32) where T
    Error_Propagator3{T, N_levels}(
        tuple([
            EPCompressor{T}([copy(zero) for _ in 1:N_arguments], false)
            for __ in 1:N_levels
        ]...),
        tuple([
            [copy(zero) for _ in 1:N_arguments]
            for __ in 1:N_levels
        ]...),
        tuple([
            [copy(zero) for _ in 1:N_arguments, __ in 1:N_arguments]
            for __ in 1:N_levels
        ]...),
        zeros(Int64, N_levels)
    )
end

function Base.push!(ep::Error_Propagator3{T, N}, args::T...) where {T, N}
    _push!(ep, 1, args)
end

function _push!(ep::Error_Propagator3{T, N}, lvl::Int64, args) where {T, N}
    # TODO Remove "..."?
    C = ep.compressors[lvl]

    # any value propagating through this function is new to lvl. Therefore we
    # add it to the sums. Note that values pushed to the output arrays are not
    # added here until the array drops to the next level. (New compressors are
    # added)
    ep.count[lvl] += 1
    # don't do any of this, it's bad
    # @views ep.sums1D[lvl] .+= args
    # @views ep.sums2D[lvl] .+= args * args'

    # do this
    @simd for i in eachindex(ep.sums1D[lvl])
        ep.sums1D[lvl][i] += args[i]
        @inbounds for j in eachindex(ep.sums1D[lvl])
            ep.sums2D[lvl][i, j] += args[i] * args[j]
        end
    end

    if !C.switch
        # Compressor has space -> save value
        C.values .= args
        C.switch = true
        return nothing
    else
        # Do averaging
        if lvl == N
            # No more propagation possible -> throw error
            throw(OverflowError("The Error Propagator has exceeded its maximum capacity."))
        else
            # propagate to next lvl
            C.switch = false
            _push!(ep, lvl+1, 0.5 * (C.values .+ args))
            return nothing
        end
    end
    return nothing


end


means(ep::Error_Propagator3, lvl) = ep.sums1D[lvl] ./ ep.count[lvl]
function covmat(ep::Error_Propagator3, lvl)
    invN = 1.0 / ep.count[lvl]
    invN1 = 1.0 / (ep.count[lvl] - 1)
    [
        (ep.sums2D[lvl][i, j] - ep.sums1D[lvl][i] * ep.sums1D[lvl][j] * invN) * invN1
        for i in eachindex(ep.sums1D[lvl]), j in eachindex(ep.sums1D[lvl])
    ]
end

function var_O1(g::Function, ep::Error_Propagator3, lvl)
    # NOTE: Not type stable :(
    # derivatives = ForwardDiff.gradient(v -> g(v...), means(ep))
    # NOTE: Also not type stable
    # NOTE: crossing a function barrier doesn"t hel either
    ms = means(ep, lvl)
    derivatives = [
        ForwardDiff.derivative(
            x -> g(ms[1:i-1]..., x, ms[i+1:end]...),
            ms[i]
        ) for i in eachindex(ms)
    ]


    result = 0.0
    invN = 1.0 / ep.count[lvl]
    invNN1 = 1.0 / (ep.count[lvl] * (ep.count[lvl] - 1))
    for i in eachindex(ep.sums1D[lvl])
        for j in eachindex(ep.sums1D[lvl])
            result += derivatives[i] * derivatives[j] * (
                ep.sums2D[lvl][i, j] - ep.sums1D[lvl][i] * ep.sums1D[lvl][j] * invN
            ) * invNN1
        end
    end

    result
end
