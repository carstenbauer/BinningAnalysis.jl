"""
    varN(BinnerA[, lvl])

Calculates the variance/N of a given level (default being the last) in the
Binning Analysis.
"""
function varN(
        B::BinnerA{N, T},
        lvl::Int64 = length(B.count)-1
    ) where {N, T <: Real}

    # lvl = 1 <=> original values
    # correct variance:
    # (∑ xᵢ^2) / (N-1) - (∑ xᵢ)(∑ xᵢ) / (N(N-1))
    (
        B.x2_sum[lvl+1] / (B.count[lvl+1] - 1) -
        B.x_sum[lvl+1]^2 / ((B.count[lvl+1] - 1) * B.count[lvl+1])
    ) / B.count[lvl+1]
end

function varN(
        B::BinnerA{N, T},
        lvl::Int64 = length(B.count)-1
    ) where {N, T <: Complex}

    # lvl = 1 <=> original values
    (
        (real(B.x2_sum[lvl+1]) + imag(B.x2_sum[lvl+1])) /
            (B.count[lvl+1] - 1) -
        (real(B.x_sum[lvl+1])^2 + imag(B.x_sum[lvl+1])^2) /
            ((B.count[lvl+1] - 1) * B.count[lvl+1])
    ) / B.count[lvl+1]
end


"""
    var(BinnerA[, lvl])

Calculates the variance of a given level (default being the last) in the
Binning Analysis.
"""
function var(
        B::BinnerA{N, T},
        lvl::Int64 = length(B.count)-1
    ) where {N, T <: Real}

    B.x2_sum[lvl+1] / (B.count[lvl+1] - 1) -
    B.x_sum[lvl+1]^2 / ((B.count[lvl+1] - 1) * B.count[lvl+1])
end

function var(
        B::BinnerA{N, T},
        lvl::Int64 = length(B.count)-1
    ) where {N, T <: Complex}

    (real(B.x2_sum[lvl+1]) + imag(B.x2_sum[lvl+1])) /
        (B.count[lvl+1] - 1) -
    (real(B.x_sum[lvl+1])^2 + imag(B.x_sum[lvl+1])^2) /
        ((B.count[lvl+1] - 1) * B.count[lvl+1])
end


"""
    all_vars(BinnerA)

Calculates the variance for each level of the Binning Analysis.
"""
function all_vars(B::BinnerA{N}) where {N}
    [var(B, lvl) for lvl in 1:N-1 if B.count[lvl+1] > 0]
end


"""
    all_varNs(BinnerA)

Calculates the variance/N for each level of the Binning Analysis.
"""
function all_varNs(B::BinnerA{N}) where {N}
    [varN(B, lvl) for lvl in 1:N-1 if B.count[lvl+1] > 0]
end


################################################################################


# NOTE works for all
"""
    mean(BinnerA[, lvl])

Calculates the mean for a given level (default being the last) in the
Binning Analysis.
"""
function mean(B::BinnerA{N}, lvl::Int64 = length(B.count)-1) where {N}
    B.x_sum[lvl+1] / B.count[lvl+1]
end


# NOTE works for all
"""
    all_means(BinnerA)

Calculates the mean for each level of the Binning Analysis.
"""
function all_means(B::BinnerA{N}) where {N}
    [mean(B, lvl) for lvl in 1:N-1 if B.count[lvl+1] > 0]
end


################################################################################


"""
    tau(BinnerA, lvl)

Calculates the autocorrelation time tau for a given binning level.
"""
function tau(B::BinnerA, lvl::Int64)
    var_0 = varN(B, 0)
    var_l = varN(B, lvl)
    0.5 * (var_l / var_0 - 1)
end


"""
    all_taus(BinnerA)

Calculates the autocorrelation time tau for each level of the Binning Analysis.
"""
function all_taus(B::BinnerA{N}) where {N}
    [tau(B, lvl) for lvl in 1:N-1 if B.count[lvl+1] > 0]
end


################################################################################


"""
    std_error(BinnerA, lvl)

Calculates the standard error for a given level.
"""
std_error(B::BinnerA, lvl::Int64) = sqrt(varN(B, lvl))


"""
    all_std_errors(BinnerA)

Calculates the standard error for each level of the Binning Analysis.
"""
all_std_errors(B::BinnerA) = map(sqrt, all_varNs(B))


"""
    convergence(BinnerA, lvl)

Computes the difference between the variance of this lvl and the last,
normalized to the last lvl. If this value tends to 0, the Binning Analysis has
converged.
"""
function convergence(B::BinnerA, lvl::Int64)
    abs((varN(B, lvl) - varN(B, lvl-1)) / varN(B, lvl-1))
end

"""
    has_converged(BinnerA, lvl[, threshhold = 0.05])

Returns true if the Binning Analysis has converged for a given lvl.
"""
function has_converged(B::BinnerA, lvl::Int64, threshhold::Float64 = 0.05)
    convergence(B, lvl) <= threshhold
end
