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


mutable struct Error_Propagator1 <: Abstract_Error_Propagator
    x_sum::Float64
    y_sum::Float64

    xx_sum::Float64
    yy_sum::Float64
    xy_sum::Float64

    N::Int64
end

# Initialization
Error_Propagator1() = Error_Propagator1(0., 0., 0., 0., 0., 0)

function Base.push!(ep::Error_Propagator1, x::Float64, y::Float64)
    # For means, variance, covariance
    ep.x_sum += x
    ep.y_sum += y
    ep.N += 1

    # Variances
    ep.xx_sum += x*x
    ep.yy_sum += y*y

    # Covariance
    ep.xy_sum += x*y
    nothing
end

# These require a final normalization
xmean(ep::Error_Propagator1) = ep.x_sum / ep.N
ymean(ep::Error_Propagator1) = ep.y_sum / ep.N
xvar(ep::Error_Propagator1) = (ep.xx_sum - ep.x_sum^2 / ep.N) / (ep.N - 1)
yvar(ep::Error_Propagator1) = (ep.yy_sum - ep.y_sum^2 / ep.N) / (ep.N - 1)
xycov(ep::Error_Propagator1) = (ep.xy_sum - ep.x_sum * ep.y_sum / ep.N) / (ep.N - 1)


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
