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
    else
        throw(ArgumentError("Keyword `method` must be either `:log`, `:full`, or `:jackknife`. Got `$(method)`."))
    end
end
