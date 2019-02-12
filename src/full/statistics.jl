# Based on https://www.amazon.com/Quantum-Monte-Carlo-Methods-Algorithms/dp/1107006422


"""
    std_error(F::FullBinner)

Estimate of the standard error of the time series mean by performing a full binning analysis.

"Full" binning means that we bin the data multiple times, considering all bin sizes compatible
with the length of the time series.
"""
function std_error end

std_error(F::FullBinner{<:Number}) = _std_error(F)
function std_error(F::FullBinner{<:AbstractArray})
    v = VectorOfArray(F)
    r = similar(F[1], real(eltype(v))) # std error is always real
    @inbounds for i in eachindex(F[1])
        r[i] = _std_error(v[i,:]) # TODO: maybe a view might be faster here
    end
    return r
end




@inline _std_error(X::AbstractVector{<:Real}) = _std_error_from_R_function(X)
@inline function _std_error(X::AbstractVector{<:Complex})
    std_err_real = _std_error_from_R_function(real(X))
    std_err_imag = _std_error_from_R_function(imag(X))
    sqrt(std_err_real^2 + std_err_imag^2)
end




"""
Estimates the error coefficient `R` by considering `<R>(bss)` (averaged R values)
and taking the largest value compatible with at least 32 bins.
"""
@inline function _std_error_from_R_function(X::AbstractVector{T}) where T<:Real
    bss, Rs, means = R_function(X; min_nbins=32)
    R = length(means) > 0 ? means[end] : 1.0
    # conv = isconverged(X, means)
    # bs = findfirst(r -> r >= R, Rs)
    return _R2std_error(X, R)
end








"""
    all_binning_errors(X) -> bs, stds, cum_stds

Returns all compatible bin sizes `bs`, the corresponding standard errors `stds`,
and the cumulative standard errors `cum_stds` (reduced fluctuations).

Only bin size that lead to at least 32 bins are considered.
"""
function all_binning_errors end

all_binning_errors(F::FullBinner{<:Number}) = _all_binning_errors(F)


function _all_binning_errors(X::AbstractVector{<:Real})
    bss, Rs, means = R_function(X; min_nbins=32)
    return bss, _R2std_error.(Ref(X), Rs), _R2std_error.(Ref(X), means)
end
function _all_binning_errors(X::AbstractVector{<:Complex})
    bss_real, stds_real, cum_stds_real = _all_binning_errors(real(X))
    bss_imag, stds_imag, cum_stds_imag = _all_binning_errors(imag(X))
    # @assert bss_real == bss_imag
    stds = sqrt.(stds_real.^2 .+ stds_imag.^2)
    cum_stds = sqrt.(cum_stds_real.^2 .+ cum_stds_imag.^2)
    return bss_real, stds, cum_stds
end











# TODO: Improve this (until then, don't export)
"""
  isconverged(X)

Checks if the estimation of the one sigma error is converged.

Returns `true` once the mean `R` value is converged up to 0.1% accuracy.
This corresponds to convergence of the error itself up to ~3% (sqrt).
"""
function isconverged(X::AbstractVector{T}) where T<:Real
  bss, Rs, means = R_function(X)
  return isconverged(X, means)
end
function isconverged(X::AbstractVector{T}, means::AbstractVector) where T<:Real
    len = length(means)
    len < 10 && (return false)
    lastn = min(len, 200)
    start = len-(lastn-1)
    return @views maximum(abs.(diff(means[start:end])))<1e-3 # convergence condition
end
isconverged(X::AbstractVector{<:Complex}) = isconverged(real(X)) && isconverged(imag(X)) # check if that's really what we want here