# see log/statistics.jl
function _reliable_level(ep::ErrorPropagator{T,N})::Int64 where {T,N}
    isempty(ep) && (return 1)                # results in NaN in std_error
    i = findlast(x -> x >= 32, ep.count)
    something(i, 1)
end

"""
    varN(ep::ErrorPropagator, i[, lvl])

Calculates the variance/N for the i-th argument of the error propagator at a
given binning level.
"""
function varN(ep::ErrorPropagator, i::Integer, lvl::Integer = _reliable_level(ep))
    n = ep.count[lvl]
    var(ep, i, lvl) / n
end

"""
    var(ep::ErrorPropagator, i[, lvl])

Calculates the variance for the i-th argument of the error propagator at a given
binning level.
"""
function var(
        ep::ErrorPropagator{T,N},
        i::Integer,
        lvl::Integer = _reliable_level(ep)
    ) where {N, T <: Real}

    n = ep.count[lvl]
    X = ep.sums1D[lvl][i]
    X2 = ep.sums2D[lvl][i, i]

    # lvl = 1 <=> original values
    # correct variance:
    # (∑ xᵢ^2) / (N-1) - (∑ xᵢ)(∑ xᵢ) / (N(N-1))
    X2 / (n - 1) - X^2 / (n*(n - 1))
end

function var(
        ep::ErrorPropagator{T,N},
        i::Integer,
        lvl::Integer = _reliable_level(ep)
    ) where {N, T <: Complex}

    n = ep.count[lvl]
    X = ep.sums1D[lvl][i]
    X2 = ep.sums2D[lvl][i, i]

    # lvl = 1 <=> original values
    (real(X2) + imag(X2)) / (n - 1) - (real(X)^2 + imag(X)^2) / (n*(n - 1))
end

function var(
        ep::ErrorPropagator{<: AbstractArray{T, D}, N},
        i::Integer,
        lvl::Integer = _reliable_level(ep)
    ) where {N, D, T <: Real}

    n = ep.count[lvl]
    X = ep.sums1D[lvl][i]
    X2 = ep.sums2D[lvl][i, i]

    @. X2 / (n - 1) - X^2 / (n*(n - 1))
end

function var(
        ep::ErrorPropagator{<: AbstractArray{T, D}, N},
        i::Integer,
        lvl::Integer = _reliable_level(ep)
    ) where {N, D, T <: Complex}

    n = ep.count[lvl]
    X = ep.sums1D[lvl][i]
    X2 = ep.sums2D[lvl][i, i]

    @. (real(X2) + imag(X2)) / (n - 1) - (real(X)^2 + imag(X)^2) / (n*(n - 1))
end

# NOTE works for all types of ErrorPropagators
"""
    mean(ep::ErrorPropagator, i[, lvl])

Calculates the mean for the i-th argument of the error propagator at a given
binning level.
"""
function mean(ep::ErrorPropagator, i::Integer, lvl::Integer = 1)
    ep.sums1D[lvl][i] / ep.count[lvl]
end

"""
    tau(ep::ErrorPropagator, i[, lvl])

Calculates the autocorrelation time tau for the i-th argument of the error
propagator at a given binning level.
"""
function tau(
        ep::ErrorPropagator{T,N},
        i::Integer,
        lvl::Integer = _reliable_level(ep)
    ) where {N , T <: Number}

    var_0 = varN(ep, i, 1)
    var_l = varN(ep, i, lvl)
    0.5 * (var_l / var_0 - 1)
end
function tau(
        ep::ErrorPropagator{T,N},
        i::Integer,
        lvl::Integer = _reliable_level(ep)
    ) where {N , T <: AbstractArray}

    var_0 = varN(ep, i, 1)
    var_l = varN(ep, i, lvl)
    @. 0.5 * (var_l / var_0 - 1)
end


"""
    std_error(ep::ErrorPropagator, i[, lvl])

Calculates the standard error of the mean for the i-th argument of the error
propagator at a given binning level.
"""
function std_error(
        ep::ErrorPropagator{T,N},
        i::Integer,
        lvl::Integer=_reliable_level(ep)
    ) where {N, T <: Number}

    sqrt(varN(ep, i, lvl))
end
function std_error(
        ep::ErrorPropagator{T,N},
        i::Integer,
        lvl::Integer=_reliable_level(ep)
    ) where {N, T <: AbstractArray}

    sqrt.(varN(ep, i, lvl))
end



# Generated functions
for name in [:varN, :var, :tau, :std_error]
    # generates functions fs(ep[, lvl]) = [f(ep, 1, lvl), .., f(ep, N_args, lvl)]
    @eval begin
        function $(Symbol(name, :s))(ep::ErrorPropagator, lvl=_reliable_level(ep))
            [$name(ep, i, lvl) for i in eachindex(ep.sums1D[1])]
        end
    end

    # generates functions
    # all_fs(ep) = [[f(ep, 1, lvl), .., f(ep, N_args, lvl)] for lvl in eachlvl]
    @eval begin
        function $(Symbol(:all_, name, :s))(ep::ErrorPropagator{T, N}) where {T, N}
            [$(Symbol(name, :s))(ep, lvl) for lvl in 1:N if ep.count[lvl] > 1]
        end
    end
end

# These should not default to _reliable_level, so keep them out of the
# function generation above
function means(ep::ErrorPropagator, lvl=1)
    [mean(ep, i, lvl) for i in eachindex(ep.sums1D[1])]
end
function all_means(ep::ErrorPropagator{T, N}) where {T, N}
    [means(ep, lvl) for lvl in 1:N if ep.count[lvl] > 1]
end

# Docs
@doc """
    varNs(ep::ErrorPropagator[, lvl])

Calculates the variance/N for each argument of the error propagator at a given
binning level.
""" varNs
@doc """
    vars(ep::ErrorPropagator[, lvl])

Calculates the variance for each argument of the error propagator at a given
binning level.
""" vars
@doc """
    means(ep::ErrorPropagator[, lvl])

Calculates the mean for each argument of the error propagator at a given
binning level.
""" means
@doc """
    std_errors(ep::ErrorPropagator[, lvl])

Calculates the standard error of the mean for each argument of the error
propagator at a given binning level.
""" std_errors


@doc """
    all_varNs(ep::ErrorPropagator)

Calculates the variance/N for each argument and binning level of the error
propagator. The result is indexed as `all_varNs(ep)[lvl][arg_idx]`.
""" all_varNs
@doc """
    all_vars(ep::ErrorPropagator)

Calculates the variance for each argument and binning level of the error
propagator. The result is indexed as `all_vars(ep)[lvl][arg_idx]`.
""" all_vars
@doc """
    all_means(ep::ErrorPropagator)

Calculates the mean for each argument and binning level of the error propagator.
The result is indexed as `all_means(ep)[lvl][arg_idx]`.
""" all_means
@doc """
    all_std_errors(ep::ErrorPropagator)

Calculates the standard error of the mean for each argument and binning level of
the error propagator. The result is indexed as `all_varNs(ep)[lvl][arg_idx]`.
""" all_std_errors


# Special Error Propagator functions


"""
    covmat(ep::ErrorPropagator[, lvl])

Returns the covariance matrix for a given level of the error propgator.
"""
function covmat(
        ep::ErrorPropagator{T, N}, lvl = _reliable_level(ep)
    ) where {N , T <: Number}

    invN = 1.0 / ep.count[lvl]
    invN1 = 1.0 / (ep.count[lvl] - 1)
    [
        (
            ep.sums2D[lvl][i, j] -
            ep.sums1D[lvl][i] * conj(ep.sums1D[lvl][j]) * invN
        ) * invN1
        for i in eachindex(ep.sums1D[lvl]), j in eachindex(ep.sums1D[lvl])
    ]
end
function covmat(
        ep::ErrorPropagator{T, N}, lvl = _reliable_level(ep)
    ) where {N , T <: AbstractArray}

    invN = 1.0 / ep.count[lvl]
    invN1 = 1.0 / (ep.count[lvl] - 1)
    [
        (
            ep.sums2D[lvl][i, j] -
            ep.sums1D[lvl][i] .* conj(ep.sums1D[lvl][j]) * invN
        ) * invN1
        for i in eachindex(ep.sums1D[lvl]), j in eachindex(ep.sums1D[lvl])
    ]
end


"""
    var(ep::ErrorPropagator, gradient[, lvl])

Gives the first-order variance estimate of a function `f` acting on the
arguments of the error propagator. `gradient` is either the gradient of `f` (a
function) or a vector `∇f(means(ep))`. To get an estimate mean value of `f`,
`mean(ep, f)` can be used.
"""
function var(
        ep::ErrorPropagator{T, N},
        gradient::Vector,
        lvl = _reliable_level(ep)
    ) where {T <: Real, N}

    result = 0.0
    invN = 1.0 / ep.count[lvl]
    # invNN1 = 1.0 / (ep.count[lvl] * (ep.count[lvl] - 1))
    invN1 = 1.0 / (ep.count[lvl] - 1)
    for i in eachindex(ep.sums1D[lvl])
        for j in eachindex(ep.sums1D[lvl])
            result += gradient[i] * gradient[j] * (
                ep.sums2D[lvl][i, j] -
                ep.sums1D[lvl][i] * ep.sums1D[lvl][j] * invN
            ) * invN1
        end
    end

    result
end
function var(
        ep::ErrorPropagator{T, N},
        gradient::Vector,
        lvl = _reliable_level(ep)
    ) where {T <: Complex, N}

    result = 0.0 * 0.0im
    invN = 1.0 / ep.count[lvl]
    # invNN1 = 1.0 / (ep.count[lvl] * (ep.count[lvl] - 1))
    invN1 = 1.0 / (ep.count[lvl] - 1)
    for i in eachindex(ep.sums1D[lvl])
        for j in eachindex(ep.sums1D[lvl])
            result += gradient[i] * conj(gradient[j]) * (
                ep.sums2D[lvl][i, j] -
                ep.sums1D[lvl][i] * conj(ep.sums1D[lvl][j]) * invN
            ) * invN1
        end
    end

    abs(result)
end

# Wrappers
function var(ep::ErrorPropagator, gradient::Function, lvl = _reliable_level(ep))
    grad = gradient(means(ep))
    if typeof(grad) <: AbstractArray
        return var(ep, grad, lvl)
    else
        return var(ep, [grad], lvl)
    end
end

"""
    varN(ep::ErrorPropagator, gradient[, lvl])

    Gives the first-order variance/N estimate of a function `f` acting on the
    arguments of the error propagator. `gradient` is either the gradient of `f` (a
    function) or a vector `∇f(means(ep))`. To get an estimate mean value of `f`,
    `mean(ep, f)` can be used.
"""
function varN(ep::ErrorPropagator, gradient::Function, lvl = _reliable_level(ep))
    var(ep, gradient, lvl) / ep.count[lvl]
end

"""
    std_error(ep::ErrorPropagator, gradient[, lvl])


Gives the first-order standard error estimate of a function `f` acting on the
arguments of the error propagator. `gradient` is either the gradient of `f` (a
function) or a vector `∇f(means(ep))`. To get an estimate mean value of `f`,
`mean(ep, f)` can be used.
"""
function std_error(ep::ErrorPropagator, gradient, lvl = _reliable_level(ep))
    sqrt(var(ep, gradient, lvl) / ep.count[lvl])
end

"""
    mean(ep, f[, lvl=1])

Returns an estimate for the mean value of `f`, where `f` is a function acting
on the sample pushed the error porpagator. `f` must be of the form `f(v)`, where
`v = [mean_arg1, mean_arg2, ..., mean_argN]` is a vector containing the averages
of each argument pushed to the error propagator.
"""
function mean(ep::ErrorPropagator, f::Function, lvl = 1)
    f(means(ep, lvl))
end
