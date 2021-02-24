"""
    varN(B::LogBinner[, lvl])

Calculates the variance/N of a given level in the Binning Analysis.
"""
function varN(B::LogBinner, lvl::Integer = _reliable_level(B))
    n = B.count[lvl]
    var(B, lvl) / n
end


"""
    var(B::LogBinner[, lvl])

Calculates the variance of a given level in the Binning Analysis.
"""
function var(B::LogBinner) end

function var(
        B::LogBinner{T,N},
        lvl::Integer = _reliable_level(B)
    ) where {N, T <: Real}

    n = B.count[lvl]
    M2 = B.x2_sum[lvl]

    # lvl = 1 <=> original values
    M2 / (n - 1)
end

function var(
        B::LogBinner{T,N},
        lvl::Integer = _reliable_level(B)
    ) where {N, T <: Complex}

    n = B.count[lvl]
    M2 = B.x2_sum[lvl]

    # lvl = 1 <=> original values
    (real(M2) + imag(M2)) / (n - 1)
end

function var(
        B::LogBinner{<: AbstractArray{T, D}, N},
        lvl::Integer = _reliable_level(B)
    ) where {N, D, T <: Real}

    n = B.count[lvl]
    M2 = B.x2_sum[lvl]

    @. M2 / (n - 1)
end

function var(
        B::LogBinner{<: AbstractArray{T, D}, N},
        lvl::Integer = _reliable_level(B)
    ) where {N, D, T <: Complex}

    n = B.count[lvl]
    M2 = B.x2_sum[lvl]

    @. (real(M2) + imag(M2)) / (n - 1)
end


"""
    all_vars(B::LogBinner)

Calculates the variance for each level of the Binning Analysis.
"""
function all_vars(B::LogBinner{T,N}) where {T,N}
    [var(B, lvl) for lvl in 1:N if B.count[lvl] > 1]
end


"""
    all_varNs(B::LogBinner)

Calculates the variance/N for each level of the Binning Analysis.
"""
function all_varNs(B::LogBinner{T,N}) where {T,N}
    [varN(B, lvl) for lvl in 1:N if B.count[lvl] > 1]
end


################################################################################


# NOTE works for all
"""
    mean(B::LogBinner[, lvl])

Calculates the mean for a given level in the Binning Analysis.
"""
function mean(B::LogBinner, lvl::Integer = 1)
    B.x_sum[lvl] / B.count[lvl]
end


# NOTE works for all
"""
    all_means(B::LogBinner)

Calculates the mean for each level of the `LogBinner`.
"""
function all_means(B::LogBinner{T,N}) where {T,N}
    [mean(B, lvl) for lvl in 1:N if B.count[lvl] > 1]
end


################################################################################


"""
    tau(B::LogBinner[, lvl])

Calculates the autocorrelation time tau.
"""
function tau(B::LogBinner{T,N}, lvl::Integer = _reliable_level(B)) where {N , T <: Number}
    var_0 = varN(B, 1)
    var_l = varN(B, lvl)
    0.5 * (var_l / var_0 - 1)
end
function tau(B::LogBinner{T,N}, lvl::Integer = _reliable_level(B)) where {N , T <: AbstractArray}
    var_0 = varN(B, 1)
    var_l = varN(B, lvl)
    @. 0.5 * (var_l / var_0 - 1)
end


"""
    all_taus(B::LogBinner)

Calculates the autocorrelation time tau for each level of the `LogBinner`.
"""
function all_taus(B::LogBinner{T,N}) where {T,N}
    [tau(B, lvl) for lvl in 1:N if B.count[lvl] > 1]
end


################################################################################


# Heuristic for selecting the level with the (presumably) most reliable
# standard error estimate:
# Take the highest lvl with at least 32 bins.
# (Chose 32 based on https://doi.org/10.1119/1.3247985)
function _reliable_level(B::LogBinner{T,N})::Int64 where {T,N}
    isempty(B) && (return 1)                # results in NaN in std_error
    i = findlast(x -> x >= 32, B.count)
    something(i, 1)
end

 """
    std_error(B::LogBinner[, lvl])

Calculates the standard error of the mean.
"""
function std_error(B::LogBinner) end

function std_error(B::LogBinner{T,N}, lvl::Integer=_reliable_level(B)) where {N, T <: Number}
    sqrt(varN(B, lvl))
end
function std_error(B::LogBinner{T,N}, lvl::Integer=_reliable_level(B)) where {N, T <: AbstractArray}
    sqrt.(varN(B, lvl))
end


"""
    all_std_errors(B::LogBinner)

Calculates the standard error for each level of the Binning Analysis.
"""
function all_std_errors(B::LogBinner) end

all_std_errors(B::LogBinner{T,N}) where {N, T <: Number} = sqrt.(all_varNs(B))
all_std_errors(B::LogBinner{T,N}) where {N, T <: AbstractArray} = (x -> sqrt.(x)).(all_varNs(B))


"""
    convergence(B::LogBinner, lvl)

Computes the difference between the variance of this lvl and the last,
normalized to the last lvl. If this value tends to 0, the Binning Analysis has
converged.
"""
function convergence(B::LogBinner) end


function convergence(B::LogBinner{T,N}, lvl::Integer=_reliable_level(B)) where {N, T <: Number}
    abs((varN(B, lvl+1) - varN(B, lvl)) / varN(B, lvl))
end
function convergence(B::LogBinner{T,N}, lvl::Integer=_reliable_level(B)) where {N, T <: AbstractArray}
    mean(abs.((varN(B, lvl+1) .- varN(B, lvl)) ./ varN(B, lvl)))
end

"""
    has_converged(B::LogBinner, lvl[, threshhold = 0.05])

Returns true if the Binning Analysis has converged for a given lvl.
"""
function has_converged(B::LogBinner, lvl::Integer=_reliable_level(B), threshhold::Float64 = 0.05)
    convergence(B, lvl) <= threshhold
end
