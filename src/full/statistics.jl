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



################################################################################
### Unbinned Statistics
################################################################################


function correlation(sample, k)
    # Stabilized correlation function χ(k) = χ(|i-j|) = ⟨xᵢxⱼ⟩ - ⟨xᵢ⟩⟨xⱼ⟩
    # Following Welford derivation from https://changyaochen.github.io/welford/

    M = length(sample)
    corr = sumx = sumy = 0.0
    for i in 1:M-k
        # ̅xᵢ₊₁ = ̅xᵢ + (xᵢ₊₁ - ̅xᵢ) / (N+1)
        # ̅yᵢ₊₁ = ̅yᵢ + (yᵢ₊₁ - ̅yᵢ) / (N+1)
        # χᵢ₊₁ = χᵢ + [(xᵢ₊₁ - ̅xᵢ)(yᵢ₊₁ - ̅yᵢ₊₁) - χᵢ] / (i + 1)

        invN = 1.0 / i
        sumy += invN * (sample[i + k] - sumy)
        sumx_delta = invN * (sample[i] - sumx)
        corr += sumx_delta * (sample[i + k] - sumy) - invN * corr
        sumx += sumx_delta
    end
    return corr
end

# QMCM eq 3.15
function unbinned_tau(sample; truncate = true, max_rel_err = 0.1, min_sample_size = 32)
    tau = 0.0

    for k in 1:length(sample) - max(1, min_sample_size)
        v = (1 - k/length(sample)) * correlation(sample, k)
        tau += v

        # We assume the worst case is a constant correlation for the tail.
        # There are (length(sample) - k) values left
        # The average prefactor is 0.5 * (1 - k/length(sample))
        # If the sum of the tail is smaller than max_rel_err * tau, i.e. if tau
        # can at most increase by a factor of max_rel_error, we stop the loop
        if truncate && 0.5 * v * (length(sample) - k) < max_rel_err * tau
            @debug "Cancelled unbinned_tau summation after $k iterations."
            break
        end
    end

    return tau / correlation(sample, 0)
end