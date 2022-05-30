# Based on Quantum Monte Carlo Methods
# https://www.amazon.com/Quantum-Monte-Carlo-Methods-Algorithms/dp/1107006422
# Chapter 3.4


################################################################################
### For generic methods
################################################################################


_reliable_level(B::FullBinner) = max(1, fld(length(B), 32))

_eachlevel(B::FullBinner) = 1:_reliable_level(B)


################################################################################
### Statistics
################################################################################



"""
    mean(B::LogBinner[, lvl])

Calculates the mean for a given level in the Binning Analysis.
"""
mean(B::FullBinner, binsize = 1) = @views mean(B.x[1:fld(end, binsize)*binsize])


"""
    var(B::LogBinner[, lvl])

Calculates the variance of a given level in the Binning Analysis.
"""
var(B::FullBinner, binsize = _reliable_level(B)) = _var(B.x, binsize)

"""
    varN(B::LogBinner[, lvl])

Calculates the variance/N of a given level in the Binning Analysis.
"""
function varN(B::FullBinner, binsize = _reliable_level(B))
    return _var(B.x, binsize) / fld(length(B.x), binsize) 
end



# This uses Welford following the implementation in log binner
function _var(A::Vector{T}, binsize) where {T <: Number}
    δ = m1 = m2 = zero(T)
    N = 0
    for i in 1 : binsize : length(A) - binsize + 1
        blockmean = @views mean(A[i : i+binsize-1])
        δ = blockmean - m1
        N += 1
        m1 += δ / N
        m2 += _prod(δ, blockmean - m1)
    end

    return __var(m2, N)
end

function _var(A::Vector{T}, binsize) where {T <: AbstractArray}
    x = A[1]
    δ  = zeros(eltype(x), size(x))
    m1 = zeros(eltype(x), size(x))
    m2 = zeros(eltype(x), size(x))
    blockmean = zeros(eltype(x), size(x))
    N = 0
    for i in 1 : binsize : length(A) - binsize + 1
        blockmean .= zero(eltype(x))
        for j in i : i+binsize-1
            blockmean .+= A[j]
        end
        blockmean ./= binsize
        @. δ = blockmean - m1
        N += 1
        @. m1 += δ / N
        @. m2 += _prod(δ, blockmean - m1)
    end

    return __var.(m2, N)
end

__var(m2::Real, N) = m2 / (N - 1)
__var(m2::Complex, N) = (real(m2) + imag(m2)) / (N - 1)