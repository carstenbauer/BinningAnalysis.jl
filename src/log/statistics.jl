################################################################################
### For generic methods
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


_eachlevel(B::LogBinner) = (i for i in 1:findlast(acc -> acc.count > 1, B.accumulators))


################################################################################
### Statistics
################################################################################


"""
    varN(B::LogBinner[, lvl])

Calculates the variance/N of a given level in the Binning Analysis.
"""
varN(B::LogBinner, lvl = _reliable_level(B)) = varN(B.accumulators[lvl])


"""
    var(B::LogBinner[, lvl])

Calculates the variance of a given level in the Binning Analysis.
"""
var(B::LogBinner, lvl = _reliable_level(B)) = var(B.accumulators[lvl])


"""
    mean(B::LogBinner[, lvl])

Calculates the mean for a given level in the Binning Analysis.
"""
mean(B::LogBinner, lvl = 1) = mean(B.accumulators[lvl])


"""
    std_error(B::LogBinner[, lvl])

Calculates the standard error of the mean.
"""
std_error(B::LogBinner, lvl = _reliable_level(B)) = std_error(B.accumulators[lvl])



################################################################################
### Experimental
################################################################################



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
