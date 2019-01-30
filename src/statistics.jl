"""
    varN(BinningAnalysis[, lvl = 1])

Calculates the variance/N of a given level in the Binning Analysis.
"""
function varN(B::LogBinner, lvl::Int64 = 1)

    n = B.count[lvl]
    var(B, lvl) / n
end


"""
    var(BinningAnalysis[, lvl = 1])

Calculates the variance of a given level in the Binning Analysis.
"""
function var(
        B::LogBinner{N, T},
        lvl::Int64 = 1
    ) where {N, T <: Real}

    n = B.count[lvl]
    X = B.x_sum[lvl]
    X2 = B.x2_sum[lvl]

    # lvl = 1 <=> original values
    # correct variance:
    # (∑ xᵢ^2) / (N-1) - (∑ xᵢ)(∑ xᵢ) / (N(N-1))
    X2 / (n - 1) - X^2 / (n*(n - 1))
end

function var(
        B::LogBinner{N, T},
        lvl::Int64 = 1
    ) where {N, T <: Complex}

    n = B.count[lvl]
    X = B.x_sum[lvl]
    X2 = B.x2_sum[lvl]

    # lvl = 1 <=> original values
    (real(X2) + imag(X2)) / (n - 1) - (real(X)^2 + imag(X)^2) / (n*(n - 1))
end

function var(
        B::LogBinner{N, <: AbstractArray{T, D}},
        lvl::Int64 = 1
    ) where {N, D, T <: Real}

    n = B.count[lvl]
    X = B.x_sum[lvl]
    X2 = B.x2_sum[lvl]

    @. X2 / (n - 1) - X^2 / (n*(n - 1))
end

function var(
        B::LogBinner{N, <: AbstractArray{T, D}},
        lvl::Int64 = 1
    ) where {N, D, T <: Complex}

    n = B.count[lvl]
    X = B.x_sum[lvl]
    X2 = B.x2_sum[lvl]

    @. (real(X2) + imag(X2)) / (n - 1) - (real(X)^2 + imag(X)^2) / (n*(n - 1))
end


"""
    all_vars(BinningAnalysis)

Calculates the variance for each level of the Binning Analysis.
"""
function all_vars(B::LogBinner{N}) where {N}
    [var(B, lvl) for lvl in 1:N if B.count[lvl] > 1]
end


"""
    all_varNs(BinningAnalysis)

Calculates the variance/N for each level of the Binning Analysis.
"""
function all_varNs(B::LogBinner{N}) where {N}
    [varN(B, lvl) for lvl in 1:N if B.count[lvl] > 1]
end


################################################################################


# NOTE works for all
"""
    mean(BinningAnalysis[, lvl=1])

Calculates the mean for a given level in the Binning Analysis.
"""
function mean(B::LogBinner, lvl::Int64 = 1)
    B.x_sum[lvl] / B.count[lvl]
end


# NOTE works for all
"""
    all_means(BinningAnalysis)

Calculates the mean for each level of the Binning Analysis.
"""
function all_means(B::LogBinner{N}) where {N}
    [mean(B, lvl) for lvl in 1:N if B.count[lvl] > 1]
end


################################################################################


"""
    tau(BinningAnalysis[, lvl=1])

Calculates the autocorrelation time tau for a given binning level.
"""
function tau(B::LogBinner, lvl::Int64 = 1)
    var_0 = varN(B, 1)
    var_l = varN(B, lvl)
    0.5 * (var_l / var_0 - 1)
end


"""
    all_taus(BinningAnalysis)

Calculates the autocorrelation time tau for each level of the Binning Analysis.
"""
function all_taus(B::LogBinner{N}) where {N}
    [tau(B, lvl) for lvl in 1:N if B.count[lvl] > 1]
end


################################################################################


# Heuristic for selecting the level with the (presumably) most reliable
# standard error estimate:
# Take the highest lvl with at least 32 bins.
# (Chose 32 based on https://doi.org/10.1119/1.3247985)
function _select_lvl_for_std_error(B::LogBinner{N,T})::Int64 where {N, T}
    isempty(B) && (return 1)                # results in NaN in std_error
    i = findlast(x -> x >= 32, B.count)
    something(i, 1)
end

 """
    std_error(BinningAnalysis[, lvl=0])

Calculates the standard error for a given level.
"""
function std_error(B::LogBinner{N, T}, lvl::Int64=_select_lvl_for_std_error(B)) where {N, T <: Number}
    sqrt(varN(B, lvl))
end

"""
    std_error(BinningAnalysis[, lvl=0])

Calculates the standard error for a given level.
"""
function std_error(B::LogBinner{N, T}, lvl::Int64=_select_lvl_for_std_error(B)) where {N, T <: AbstractArray}
    sqrt.(varN(B, lvl))
end


"""
    all_std_errors(BinningAnalysis)

Calculates the standard error for each level of the Binning Analysis.
"""
function all_std_errors(B::LogBinner{N, T}) where {N, T <: Number}
    sqrt.(all_varNs(B))
end
function all_std_errors(B::LogBinner{N, T}) where {N, T <: AbstractArray}
    (x -> sqrt.(x)).all_varNs(B)
end


"""
    convergence(BinningAnalysis, lvl)

Computes the difference between the variance of this lvl and the last,
normalized to the last lvl. If this value tends to 0, the Binning Analysis has
converged.
"""
function convergence(B::LogBinner{N, T}, lvl::Int64) where {N, T <: Number}
    abs((varN(B, lvl+1) - varN(B, lvl)) / varN(B, lvl))
end
function convergence(B::LogBinner{N, T}, lvl::Int64) where {N, T <: AbstractArray}
    mean(abs.((varN(B, lvl+1) .- varN(B, lvl)) ./ varN(B, lvl)))
end

"""
    has_converged(BinningAnalysis, lvl[, threshhold = 0.05])

Returns true if the Binning Analysis has converged for a given lvl.
"""
function has_converged(B::LogBinner, lvl::Int64, threshhold::Float64 = 0.05)
    convergence(B, lvl) <= threshhold
end
