module BinningAnalysis

import Base: push!, mean, var


# This is a Node in the Binning Analysis tree that averages two values. There
# is one of these for each binning level. When two values should be
# compressed, this is done immediately, so that only one value needs to be saved.
# switch indicates whether value should be written to or averaging should happen.
mutable struct Compressor
    value::Float64
    switch::UInt8
end


struct BinnerA{N}
    # list of Compressors, one per level
    compressors::NTuple{N, Compressor}

    # sum(x) for all values on a given lvl
    x_sum::Vector{Float64}
    # sum(x.^2) for all values on a given lvl
    x2_sum::Vector{Float64}
    # number of values that are summed on a given lvl
    count::Vector{Int64}
end


"""
    BinnerA([, N = 32])

Creates a new Binning Analysis which can take 2^(N-1) values. Returns a Binning Analysis object. Use push! to add values.
"""
function BinnerA(N::Int64 = 32)
    BinnerA{N}(
        ([Compressor(0.0, 0) for i in 1:N]...),
        zeros(Float64, N),
        zeros(Float64, N),
        zeros(Int64, N)
    )
end


"""
    push!(BinningAnalysis, value)

Pushes a new value into the Binning Analysis.
"""
function push!(B::BinnerA, value::Float64)
    push!(B, 1, value)
end


# recursion, back-end function
function push!(B::BinnerA{N}, lvl::Int64, value::Float64) where {N}
    C = B.compressors[lvl]

    # any value propagating through this function is new to lvl. Therefore we
    # add it to the sums. Note that values pushed to the output arrays are not
    # added here until the array drops to the next level. (New compressors are
    # added)
    B.x_sum[lvl] += value
    B.x2_sum[lvl] += value^2
    B.count[lvl] += 1

    if C.switch == 0
        # Compressor has space -> save value
        C.value = value
        C.switch = 1
        return nothing
    else
        # Do averaging
        if lvl == N
            # No more propagation possible -> throw error
            error("The Binning Analysis ha exceeddd its maximum capacity.")
        else
            # propagate to next lvl
            C.switch = 0
            push!(B, lvl+1, 0.5 * (C.value + value))
            return nothing
        end
    end
    return nothing
end



"""
    varN(BinningAnalysis[, lvl])

Calculates the variance/N of a given level (default being the last) in the Binning Analysis.
"""
function varN(B::BinnerA, lvl::Int64=length(B.count)-1)
    # lvl = 1 <=> original values
    return (
        (B.x2_sum[lvl+1] / B.count[lvl+1]) -
        (B.x_sum[lvl+1] / B.count[lvl+1])^2
    ) / (B.count[lvl+1] - 1)
end


"""
    var(BinningAnalysis[, lvl])

Calculates the variance of a given level (default being the last) in the Binning Analysis.
"""
function var(B::BinnerA, lvl::Int64=length(B.count)-1)
    (B.x2_sum[lvl+1] / B.count[lvl+1]) - (B.x_sum[lvl+1] / B.count[lvl+1])^2
end


"""
    all_vars(BinningAnalysis)

Calculates the variance for each level of the Binning Analysis.
"""
function all_vars(B::BinnerA{N}) where {N}
    [var(B, lvl) for lvl in 1:N-1 if B.count[lvl+1] > 0]
end


"""
    all_varNs(BinningAnalysis)

Calculates the variance/N for each level of the Binning Analysis.
"""
function all_varNs(B::BinnerA{N}) where {N}
    [varN(B, lvl) for lvl in 1:N-1 if B.count[lvl+1] > 0]
end


"""
    mean(BinningAnalysis[, lvl])

Calculates the mean for a given level (default being the last) in the Binning Analysis
"""
function mean(B::BinnerA, lvl::Int64=length(B.count)-1)
    return B.x_sum[lvl+1] / B.count[lvl+1]
end


"""
    all_means(BinningAnalysis)

Calculates the mean for each level of the Binning Analysis.
"""
function all_means(B::BinnerA{N}) where {N}
    [mean(B, lvl) for lvl in 1:N-1 if B.count[lvl+1] > 0]
end


"""
    tau(BinningAnalysis, lvl)

Calculates the autocorrelation time tau for a given binning level.
"""
function tau(B::BinnerA, lvl::Int64)
    var_0 = var(B, 0)
    var_l = var(B, lvl)
    0.5 * (var_l / var_0 - 1)
end


"""
    all_taus(BinningAnalysis)

Calculates the autocorrelation time tau for each level of the Binning Analysis.
"""
function all_vars(B::BinnerA{N}) where {N}
    [tau(B, lvl) for lvl in 1:N-1 if B.count[lvl+1] > 0]
end

"""
    std_error(BinningAnalysis, lvl)

Calculates the standard error for a given level.
"""
std_error(B::BinnerA, lvl::Int64) = sqrt(varN(B, lvl)


"""
    all_std_errors(BinningAnalysis)

Calculates the standard error for each level of the Binning Analysis.
"""
all_std_errors(B::BinnerA) = map(sqrt, all_varNs(B))


"""
    convergence(BinningAnalysis, lvl)

Computes the difference between the variance of this lvl and the last, normalized
to the last lvl. If this value tends to 0, the Binning Analysis has converged.
"""
function convergence(B::BinnerA, lvl::Int64)
    abs((varN(B, lvl) - varN(B, lvl-1)) / varN(B, lvl-1))
end

"""
    has_converged(BinningAnalysis, lvl[, threshhold = 0.05])

Returns true if the Binning Analysis has converged for a given lvl.
"""
function has_converged(B::BinnerA, lvl::Int64, threshhold::Float64 = 0.05)
    convergence(B, lvl) <= threshhold
end


export BinnerA, push!
export mean, var, varN, tau
export all_means, all_vars, all_varNs, all_taus
export convergence, has_converged

end # module
