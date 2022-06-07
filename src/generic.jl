abstract type AbstractBinner{T} end

Base.eltype(::AbstractBinner{T}) where {T} = T

"""
    append!(B::AbstractBinner, values::AbstractArray)

Adds an array of values to the binner.
"""
function Base.append!(B::AbstractBinner, values::AbstractArray)
    for v in values
        push!(B, v)
    end
    return
end


################################################################################
### all_x methods
################################################################################

# _eachlevel is a bit weird...
# For LogBinner and ErrorPropagator a level would be binsize = 2^lvl
# For FullBinner (and similar) it would refer to binsizes


"""
    all_vars(B::AbstractBinner)

Calculates the variance for each binning level of a given binner. 
"""
all_vars(B::AbstractBinner) = [var(B, lvl) for lvl in _eachlevel(B)]


"""
    all_varNs(B::AbstractBinner)

Calculates the variance/N for each binning level of a given binner.
"""
all_varNs(B::AbstractBinner) = [varN(B, lvl) for lvl in _eachlevel(B)]

"""
    all_means(B::AbstractBinner)

Calculates the mean for each binning level of a given binner.
"""
all_means(B::AbstractBinner) = [mean(B, lvl) for lvl in _eachlevel(B)]

"""
    all_taus(B::AbstractBinner)

Calculates the autocorrelation time tau for each binning level of a given binner.
"""
all_taus(B::AbstractBinner) = [tau(B, lvl) for lvl in _eachlevel(B)]

"""
    all_autocorrelations(B::AbstractBinner)

Calculates the autocorrelation time tau for each binning level of a given binner.
"""
all_autocorrelations(B::AbstractBinner) = all_taus(B)

"""
    all_autocorrelation_times(B::AbstractBinner)

Calculates the autocorrelation time tau for each binning level of a given binner.
"""
all_autocorrelation_times(B::AbstractBinner) = all_taus(B)


"""
    all_std_errors(B::AbstractBinner)

Calculates the standard error for each binning level of a given binner.
"""
all_std_errors(B::AbstractBinner) = [std_error(B, lvl) for lvl in _eachlevel(B)]



################################################################################
### Generic Statistics code
################################################################################



"""
    std_error(B::AbstractBinner[, lvl])

Calculates the standard error of the mean.
"""
function std_error(B::AbstractBinner{<: Number}, lvl = _reliable_level(B))
    return sqrt(max(0, varN(B, lvl)))
end

function std_error(B::AbstractBinner{<: AbstractArray}, lvl = _reliable_level(B))
    return sqrt.(max.(0, varN(B, lvl)))
end



################################################################################
### Autocorrelation time
################################################################################

# Ref: Quantum Monte Carlo Methods, Eq. 3.21, Chapter 3.4
# Note: 
# varN returns the variance / number of elements for a given level
# The reference uses m = size of bins 
#               = total number of elements / number of elements at a given level

"""
    autocorrelation(B::AbstractBinner[, lvl])

Calculates the autocorrelation time tau for a given binner relative to an 
optional binning level. The default binning level is picked such that at least 
32 bins exist.
"""
autocorrelation(B::AbstractBinner, lvl = _reliable_level(B)) = tau(B, lvl)

"""
    autocorrelation_time(B::AbstractBinner[, lvl])

Calculates the autocorrelation time tau for a given binner relative to an 
optional binning level. The default binning level is picked such that at least 
32 bins exist.
"""
autocorrelation_time(B::AbstractBinner, lvl = _reliable_level(B)) = tau(B, lvl)

"""
    tau(B::LogBinner[, lvl])

Calculates the autocorrelation time tau for a given binner relative to an 
optional binning level. The default binning level is picked such that at least 
32 bins exist.
"""
function tau(B::AbstractBinner{<: Number}, lvl = _reliable_level(B))
    return 0.5 * (varN(B, lvl) / varN(B, 1) - 1)
end
function tau(B::AbstractBinner{<: AbstractArray}, lvl = _reliable_level(B))
    return 0.5 .* (varN(B, lvl) ./ varN(B, 1) .- 1)
end



################################################################################
### Wrappers
################################################################################



"""
    std_error(x[; method])

Estimate the standard error of the mean of the given time series.

The keyword `method` can be used to choose one out of the following methods:

* Logarithmic binning: `:log` (default)
* Full binning: `:full`
* Jackknife resampling: `:jackknife`
"""
function std_error(x::AbstractVector{T}; method::Symbol=:log) where T
    if method == :log
        B = LogBinner(zero(x[1]), capacity=length(x))
        append!(B, x)
        return std_error(B)
    elseif method == :full
        return std_error(FullBinner(x))
    elseif method == :jackknife
        T <: Number || error("Jackknife doesn't support non-number type time series.")
        return Jackknife.std_error(identity, x)
    elseif method == :error_propagator
        B = ErrorPropagator(zero(x[1]), capacity=length(x))
        append!(B, x)
        return std_error(B, 1)
    else
        throw(ArgumentError(
            "Keyword `method` must be either `:log`, `:full`, `:jackknife` or " *
            "`:error_propagator`. Got `$(method)`."
        ))
    end
end


"""
    tau(x[; method])

Estimate the autocorrelation time of the given time series.

The keyword `method` can be used to choose one out of the following methods:

* Logarithmic binning: `:log` (default)
* Full binning: `:full`
"""
function tau(x::AbstractVector{T}; method::Symbol=:log) where T
    if method == :log
        B = LogBinner(zero(x[1]), capacity=length(x))
        append!(B, x)
        return tau(B)
    elseif method == :full
        return tau(FullBinner(x))
    else
        throw(ArgumentError("Keyword `method` must be either `:log` or `:full`. Got `$(method)`."))
    end
end
