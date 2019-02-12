# Based on https://www.amazon.com/Quantum-Monte-Carlo-Methods-Algorithms/dp/1107006422


"""
    binning_error(X)

Estimate of the standard error of the time series mean by performing a full binning analysis.

"Full" binning means that we bin the data multiple times, considering all bin sizes compatible
with the length of `X`.
"""
function binning_error end

binning_error(X::AbstractVector{<:Number}) = _binning_error(X)
binning_error(X::AbstractArray{<:Number}) = begin nd = ndims(X); dropdims(mapslices(xi->binning_error(xi), X, dims=nd), dims=nd) end
binning_error(X::AbstractVector{<:AbstractArray}) = binning_error(cat(X..., dims=ndims(X[1])+1))




@inline _binning_error(X::AbstractVector{<:Real}) = _std_error_from_R_function(X)
@inline function _binning_error(X::AbstractVector{<:Complex})
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

function all_binning_errors(X::AbstractVector{<:Real})
    bss, Rs, means = R_function(X; min_nbins=32)
    return bss, _R2std_error.(Ref(X), Rs), _R2std_error.(Ref(X), means)
end
function all_binning_errors(X::AbstractVector{<:Complex})
    bss_real, stds_real, cum_stds_real = all_binning_errors(real(X))
    bss_imag, stds_imag, cum_stds_imag = all_binning_errors(imag(X))
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














#####
# Calculation of error coefficient (function) R. (Ch. 3.4 in QMC book)
#####
"""
Groups datapoints in bins of fixed binsize and returns error coefficient R. (Eq. 3.20)
"""
function R_value(X::AbstractVector{T}, binsize::Int) where T<:Real
    N = length(X)
    n_bins = div(N,binsize)
    lastbs = rem(N,binsize)

    @views blockmeans = vec(mean(reshape(X[1:n_bins*binsize], (binsize,n_bins)), dims=1))
    # if lastbs != 0
    #     vcat(blockmeans, mean(X[n_bins*binsize+1:end]))
    #     n_bins += 1
    # end

    blocksigma2 = 1/(n_bins-1)*sum((blockmeans .- mean(X)).^2)
    return binsize * blocksigma2 / var(X)
end


_R2std_error(X, Rvalue) = sqrt(Rvalue*var(X)/length(X))


"""
Groups datapoints in bins of varying size `bs`.
Returns the used binsizes `bss`, the error coefficient function values `R(bss)` (Eq. 3.20), and 
the cumulative error coefficients `<R>(bss)`. The function should feature a plateau, 
i.e. `R(bs_p) ~ R(bs)` for `bs >= bs_p`. (Fig. 3.3)

Optional keyword `min_nbins`. Only bin sizes used that lead to at least `min_nbins` bins.
"""
function R_function(X::AbstractVector{T}; min_nbins=32) where T<:Real
    max_binsize = floor(Int, length(X)/min_nbins)
    binsizes = 1:max_binsize

    R = Vector{Float64}(undef, length(binsizes))
    @inbounds for bs in binsizes
        R[bs] = R_value(X, bs)
    end

    means = @views mean.([R[1:i] for i in 1:max_binsize])
    return binsizes, R, means
end