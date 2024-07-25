################################################################################
### For generic methods
################################################################################


# Heuristic for selecting the level with the (presumably) most reliable
# standard error estimate:
# Take the highest lvl with at least 32 bins.
# (Chose 32 based on https://doi.org/10.1119/1.3247985)
function _reliable_level(B::ErrorPropagator{T,N})::Int where {T,N}
    isempty(B) && (return 1)                # results in NaN in std_error
    i = findlast(x -> x >= 32, B.count)
    return something(i, 1)
end


_eachlevel(B::ErrorPropagator) = 1:findlast(x -> x > 1, B.count)


################################################################################
### Statistics
################################################################################



"""
    mean(B::ErrorPropagator, i[, lvl = 1])

Calculates the mean for the i-th argument of the error propagator at an optional
binning level.
"""
function mean(B::ErrorPropagator, i::Integer, lvl = 1)
    return B.sums1D[lvl][i] / B.count[lvl]
end



"""
    var(B::ErrorPropagator, i[, lvl])

Calculates the variance for the i-th argument of the error propagator at a given
binning level.
"""
function var(B::ErrorPropagator{<: Real}, i::Integer, lvl = _reliable_level(B))
    n = B.count[lvl]
    X = B.sums1D[lvl][i]
    X2 = B.sums2D[lvl][i, i]

    # lvl = 1 <=> original values
    # correct variance:
    # (∑ xᵢ^2) / (N-1) - (∑ xᵢ)(∑ xᵢ) / (N(N-1))
    return X2 / (n - 1) - X^2 / (n*(n - 1))
end

function var(B::ErrorPropagator{<: Complex}, i::Integer, lvl = _reliable_level(B))
    n = B.count[lvl]
    X = B.sums1D[lvl][i]
    X2 = B.sums2D[lvl][i, i]

    return (real(X2) + imag(X2)) / (n - 1) - (real(X)^2 + imag(X)^2) / (n*(n - 1))
end

function var(B::ErrorPropagator{<: AbstractArray{<: Real}}, i::Integer, lvl = _reliable_level(B))
    n = B.count[lvl]
    X = B.sums1D[lvl][i]
    X2 = B.sums2D[lvl][i, i]

    return @. X2 / (n - 1) - X^2 / (n*(n - 1))
end

function var(B::ErrorPropagator{<: AbstractArray{<: Complex}}, i::Integer, lvl = _reliable_level(B))
    n = B.count[lvl]
    X = B.sums1D[lvl][i]
    X2 = B.sums2D[lvl][i, i]

    return @. (real(X2) + imag(X2)) / (n - 1) - (real(X)^2 + imag(X)^2) / (n*(n - 1))
end


"""
    varN(B::ErrorPropagator, i[, lvl])

Calculates the variance/N for the i-th argument of the error propagator at a
given binning level.
"""
function varN(B::ErrorPropagator, i::Integer, lvl = _reliable_level(B))
    return var(B, i, lvl) / B.count[lvl]
end



"""
    tau(B::ErrorPropagator, i[, lvl])

Calculates the autocorrelation time tau for the i-th argument of the error
propagator at a given binning level.
"""
function tau(B::ErrorPropagator{<: Number}, i::Integer, lvl = _reliable_level(B))
    return 0.5 * (varN(B, i, lvl) / varN(B, i, 1) - 1)
end
function tau(B::ErrorPropagator{<: AbstractArray}, i::Integer, lvl = _reliable_level(B))
    return 0.5 * (varN(B, i, lvl) ./ varN(B, i, 1) .- 1)
end
autocorrelation(B::ErrorPropagator, i::Integer, lvl = _reliable_level(B)) = tau(B, i, lvl)
autocorrelation_time(B::ErrorPropagator, i::Integer, lvl = _reliable_level(B)) = tau(B, i, lvl)



"""
    std_error(B::ErrorPropagator, i[, lvl])

Calculates the standard error of the mean for the i-th argument of the error
propagator at a given binning level.
"""
function std_error(B::ErrorPropagator{<: Number}, i::Integer, lvl=_reliable_level(B))
    return sqrt(varN(B, i, lvl))
end
function std_error(B::ErrorPropagator{<: AbstractArray}, i::Integer, lvl=_reliable_level(B))
    return sqrt.(varN(B, i, lvl))
end


################################################################################
### Additional methods
################################################################################


for name in [:mean, :varN, :var, :tau, :std_error, :autocorrelation, :autocorrelation_time]
    # generates functions fs(B[, lvl]) = [f(B, 1, lvl), .., f(B, N_args, lvl)]
    @eval begin
        function $(Symbol(name, :s))(B::ErrorPropagator, lvl=_reliable_level(B))
            return [$name(B, i, lvl) for i in eachindex(B.sums1D[1])]
        end
    end

    # all_fs(ep) = [[f(ep, 1, lvl), .., f(ep, N_args, lvl)] for lvl in eachlvl]
    @eval begin
        function $(Symbol(:all_, name, :s))(B::ErrorPropagator)
            return [$(Symbol(name, :s))(B, lvl) for lvl in _eachlevel(B)]
        end
    end
end


# Docs
@doc """
    varNs(B::ErrorPropagator[, lvl])

Calculates the variance/N for each argument of the error propagator at a given
binning level.
""" varNs
@doc """
    vars(B::ErrorPropagator[, lvl])

Calculates the variance for each argument of the error propagator at a given
binning level.
""" vars
@doc """
    means(B::ErrorPropagator[, lvl])

Calculates the mean for each argument of the error propagator at a given
binning level.
""" means
@doc """
    std_errors(B::ErrorPropagator[, lvl])

Calculates the standard error of the mean for each argument of the error
propagator at a given binning level.
""" std_errors


"""
    covmat(B::ErrorPropagator[, lvl])

Returns the covariance matrix for a given level of the error propgator.
"""
function covmat(
        B::ErrorPropagator{T, N}, lvl = _reliable_level(B)
    ) where {N , T <: Number}

    invN = 1.0 / B.count[lvl]
    invN1 = 1.0 / (B.count[lvl] - 1)
    return [
        (
            B.sums2D[lvl][i, j] -
            B.sums1D[lvl][i] * conj(B.sums1D[lvl][j]) * invN
        ) * invN1
        for i in eachindex(B.sums1D[lvl]), j in eachindex(B.sums1D[lvl])
    ]
end
function covmat(
        B::ErrorPropagator{T, N}, lvl = _reliable_level(B)
    ) where {N , T <: AbstractArray}

    invN = 1.0 / B.count[lvl]
    invN1 = 1.0 / (B.count[lvl] - 1)
    return [
        (
            B.sums2D[lvl][i, j] -
            B.sums1D[lvl][i] .* conj(B.sums1D[lvl][j]) * invN
        ) * invN1
        for i in eachindex(B.sums1D[lvl]), j in eachindex(B.sums1D[lvl])
    ]
end


"""
    var(B::ErrorPropagator, gradient[, lvl])

Gives the first-order variance estimate of a function `f` acting on the
arguments of the error propagator. `gradient` is either the gradient of `f` (a
function) or a vector `∇f(means(B))`. To get an estimate mean value of `f`,
`mean(B, f)` can be used.
"""
function var(
        B::ErrorPropagator{T, N},
        gradient::Vector,
        lvl = _reliable_level(B)
    ) where {T <: Real, N}

    result = 0.0
    invN = 1.0 / B.count[lvl]
    # invNN1 = 1.0 / (B.count[lvl] * (B.count[lvl] - 1))
    invN1 = 1.0 / (B.count[lvl] - 1)
    for i in eachindex(B.sums1D[lvl])
        for j in eachindex(B.sums1D[lvl])
            result += gradient[i] * gradient[j] * (
                B.sums2D[lvl][i, j] -
                B.sums1D[lvl][i] * B.sums1D[lvl][j] * invN
            ) * invN1
        end
    end

    return result
end

function var(
        B::ErrorPropagator{T, N},
        gradient::Vector,
        lvl = _reliable_level(B)
    ) where {T <: Complex, N}

    result = 0.0 * 0.0im
    invN = 1.0 / B.count[lvl]
    # invNN1 = 1.0 / (B.count[lvl] * (B.count[lvl] - 1))
    invN1 = 1.0 / (B.count[lvl] - 1)
    for i in eachindex(B.sums1D[lvl])
        for j in eachindex(B.sums1D[lvl])
            result += gradient[i] * conj(gradient[j]) * (
                B.sums2D[lvl][i, j] -
                B.sums1D[lvl][i] * conj(B.sums1D[lvl][j]) * invN
            ) * invN1
        end
    end

    return abs(result)
end

# Wrappers
function var(B::ErrorPropagator, gradient::Function, lvl = _reliable_level(B))
    grad = gradient(means(B))
    if typeof(grad) <: AbstractArray
        return var(B, grad, lvl)
    else
        return var(B, [grad], lvl)
    end
end

"""
    varN(B::ErrorPropagator, gradient[, lvl])

    Gives the first-order variance/N estimate of a function `f` acting on the
    arguments of the error propagator. `gradient` is either the gradient of `f` (a
    function) or a vector `∇f(means(B))`. To get an estimate mean value of `f`,
    `mean(B, f)` can be used.
"""
function varN(B::ErrorPropagator, gradient::Function, lvl = _reliable_level(B))
    return var(B, gradient, lvl) / B.count[lvl]
end

"""
    std_error(B::ErrorPropagator, gradient[, lvl])


Gives the first-order standard error estimate of a function `f` acting on the
arguments of the error propagator. `gradient` is either the gradient of `f` (a
function) or a vector `∇f(means(B))`. To get an estimate mean value of `f`,
`mean(B, f)` can be used.
"""
function std_error(B::ErrorPropagator, gradient::Function, lvl = _reliable_level(B))
    return sqrt(var(B, gradient, lvl) / B.count[lvl])
end

"""
    mean(B, f[, lvl=1])

Returns an estimate for the mean value of `f`, where `f` is a function acting
on the sample pushed the error porpagator. `f` must be of the form `f(v)`, where
`v = [mean_arg1, mean_arg2, ..., mean_argN]` is a vector containing the averages
of each argument pushed to the error propagator.
"""
function mean(B::ErrorPropagator, f::Function, lvl = 1)
    return f(means(B, lvl))
end
