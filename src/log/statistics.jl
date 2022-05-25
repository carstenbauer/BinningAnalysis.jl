"""
    varN(B::LogBinner[, lvl])

Calculates the variance/N of a given level in the Binning Analysis.
"""
function varN(B::LogBinner, lvl = _reliable_level(B))
    return varN(B.accumulators[lvl])
end


"""
    var(B::LogBinner[, lvl])

Calculates the variance of a given level in the Binning Analysis.
"""
function var(B::LogBinner{T,N}, lvl = _reliable_level(B)) where {N, T}
    return var(B.accumulators[lvl])
end


"""
    all_vars(B::LogBinner)

Calculates the variance for each level of the Binning Analysis.
"""
function all_vars(B::LogBinner{T,N}) where {T,N}
    return [var(B, lvl) for lvl in 1:N if count(B, lvl) > 1]
end


"""
    all_varNs(B::LogBinner)

Calculates the variance/N for each level of the Binning Analysis.
"""
function all_varNs(B::LogBinner{T,N}) where {T,N}
    return [varN(B, lvl) for lvl in 1:N if count(B, lvl) > 1]
end


################################################################################


# NOTE works for all
"""
    mean(B::LogBinner[, lvl])

Calculates the mean for a given level in the Binning Analysis.
"""
mean(B::LogBinner, lvl = 1) = mean(B.accumulators[lvl])


# NOTE works for all
"""
    all_means(B::LogBinner)

Calculates the mean for each level of the `LogBinner`.
"""
function all_means(B::LogBinner{T,N}) where {T,N}
    return [mean(B, lvl) for lvl in 1:N if count(B, lvl) > 1]
end


################################################################################
# Ref: Quantum Monte Carlo Methods, Eq. 3.21, Chapter 3.4
# Note: 
# varN returns the variance / number of elements for a given level
# The reference uses m = size of bins 
#               = total number of elements / number of elements at a given level


"""
    tau(B::LogBinner[, lvl])

Calculates the autocorrelation time tau.
"""
function tau(B::LogBinner{T,N}, lvl = _reliable_level(B)) where {N , T <: Number}
    var_0 = varN(B, 1)
    var_l = varN(B, lvl)
    return 0.5 * (var_l / var_0 - 1)
end
function tau(B::LogBinner{T,N}, lvl = _reliable_level(B)) where {N , T <: AbstractArray}
    var_0 = varN(B, 1)
    var_l = varN(B, lvl)
    return @. 0.5 * (var_l / var_0 - 1)
end


"""
    all_taus(B::LogBinner)

Calculates the autocorrelation time tau for each level of the `LogBinner`.
"""
function all_taus(B::LogBinner{T,N}) where {T,N}
    return [tau(B, lvl) for lvl in 1:N if count(B, lvl) > 1]
end


################################################################################


# Heuristic for selecting the level with the (presumably) most reliable
# standard error estimate:
# Take the highest lvl with at least 32 bins.
# (Chose 32 based on https://doi.org/10.1119/1.3247985)
function _reliable_level(B::LogBinner{T,N})::Int64 where {T,N}
    isempty(B) && (return 1)                # results in NaN in std_error
    i = findlast(x -> x.count >= 32, B.accumulators)
    return something(i, 1)
end

"""
    std_error(B::LogBinner[, lvl])

Calculates the standard error of the mean.
"""
function std_error(B::LogBinner{T,N}, lvl = _reliable_level(B)) where {N, T}
    return std_error(B.accumulators[lvl])
end

"""
    all_std_errors(B::LogBinner)

Calculates the standard error for each level of the Binning Analysis.
"""
function all_std_errors(B::LogBinner{T,N}) where {N, T}
    return [std_error(B, lvl) for lvl in 1:N if count(B, lvl) > 1]
end


"""
    convergence(B::LogBinner, lvl)

Computes the difference between the variance of this lvl and the last,
normalized to the last lvl. If this value tends to 0, the Binning Analysis has
converged.
"""
function convergence(B::LogBinner) end


function convergence(B::LogBinner{T,N}, lvl = _reliable_level(B)) where {N, T <: Number}
    return abs((varN(B, lvl+1) - varN(B, lvl)) / varN(B, lvl))
end
function convergence(B::LogBinner{T,N}, lvl = _reliable_level(B)) where {N, T <: AbstractArray}
    return mean(abs.((varN(B, lvl+1) .- varN(B, lvl)) ./ varN(B, lvl)))
end

"""
    has_converged(B::LogBinner, lvl[, threshhold = 0.05])

Returns true if the Binning Analysis has converged for a given lvl.
"""
function has_converged(B::LogBinner, lvl = _reliable_level(B), threshhold::Float64 = 0.05)
    return convergence(B, lvl) <= threshhold
end
