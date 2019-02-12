#####
# Calculation of error coefficient (function) R. (Ch. 3.4 in QMC book)
#####
"""
Groups datapoints in bins of varying size `bs`.
Returns the used binsizes `bss`, the error coefficient function `R(bss)` (Eq. 3.20), and 
the averaged error coefficients `<R>(bss)`. The function should feature a plateau, 
i.e. `R(bs_p) ~ R(bs)` for `bs >= bs_p`. (Fig. 3.3)

Optional keyword `min_nbins`. Only bin sizes used that lead to at least `min_nbins` bins.
"""
function R_function(X::AbstractVector{T}; min_nbins=50) where T<:Real
    max_binsize = floor(Int, length(X)/min_nbins)
    binsizes = 1:max_binsize

    R = Vector{Float64}(undef, length(binsizes))
    @inbounds for bs in binsizes
        R[bs] = R_value(X, bs)
    end

    means = @views mean.([R[1:i] for i in 1:max_binsize])
    return binsizes, R, means
end

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


#####
# Automatic R/error estimation + convergence check
#####
"""
Estimates the error coefficient `R` by considering `<R>(bss)` (averaged R values).

Returns estimated start of plateau (bin size), estimate for `R`, and convergence (boolean).
"""
function Rplateaufinder(X::AbstractVector{T}) where T<:Real
    bss, Rs, means = R_function(X)

    conv = isconverged(X, means)
    # R = maximum(means[(end-(lastn-1)):end])
    R = length(means)>0 ? means[end] : 1.0
    bs = findfirst(r->r>=R, Rs)

    return bs, R, conv
end

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
# Calculation of binning error (automatic plateau finder, automatic naive (last value), from given R value, for given bin size)
#####
"""
    binning_error(X)

Estimate of the one-sigma error of the time series's mean.
Respects correlations between measurements through binning analysis.

Note that this is not the same as `Base.std(X)`, not even
for uncorrelated measurements.

For more details, see [`MonteCarloObservable.Rplateaufinder`](@ref).
"""
binning_error(X::AbstractVector{<:Number}) = binning_error_with_convergence(X)[1]
binning_error(X::AbstractArray{<:Number}) = begin nd = ndims(X); dropdims(mapslices(xi->binning_error(xi), X, dims=nd), dims=nd) end
binning_error(X::AbstractVector{<:AbstractArray}) = binning_error(cat(X..., dims=ndims(X[1])+1))

"""
Returns one sigma error and convergence flag (boolean).
"""
function binning_error_with_convergence(X::AbstractVector{T}) where T<:Real
  bs, R, conv = Rplateaufinder(X)
  binning_error_from_R(X, R), conv
end
function binning_error_with_convergence(X::AbstractVector{T}) where T<:Complex
    Xreal = real(X)
    bsreal, Rreal, convreal = Rplateaufinder(Xreal)
    Ximag = imag(X)
    bsimag, Rimag, convimag = Rplateaufinder(Ximag)
    sqrt(binning_error_from_R(Xreal, Rreal)^2 + binning_error_from_R(Xreal, Rimag)^2), convreal && convimag # check if that's really what we want here
end
binning_error_with_convergence(X::AbstractArray{<:Number}) = begin nd = ndims(X); errconv = dropdims(mapslices(xi->binning_error_with_convergence(xi), X, dims=nd), dims=nd); getindex.(errconv, 1), getindex.(errconv, 2) end
binning_error_with_convergence(X::AbstractVector{<:AbstractArray}) = binning_error_with_convergence(cat(X..., dims=ndims(X[1])+1))

"""
    binning_error(X, binsize)

Estimate of the one-sigma error of the time series's mean.
Respect correlations between measurements through binning analysis, 
using the given `binsize` (i.e. assuming independence of bins, Eq. 3.18 basically).
"""
binning_error(X::AbstractVector{<:Real}, binsize::Int) = binning_error_from_R(X, R_value(X, binsize))
binning_error(X::AbstractVector{<:Complex}, binsize::Int) = sqrt(binning_error(real(X), binsize)^2 + binning_error(imag(X), binsize)^2)
binning_error(X::AbstractArray{<:Number}, binsize::Int) = begin nd = ndims(X); dropdims(mapslices(xi->binning_error(xi, binsize), X, dims=nd), dims=nd) end
binning_error(X::AbstractVector{<:AbstractArray}, binsize::Int) = binning_error(cat(X..., dims=ndims(X[1])+1), binsize)


binning_error_from_R(X::AbstractVector{T}, Rvalue::Float64) where T<:Real = sqrt(Rvalue*var(X)/length(X))

"""
    binning_error_naive(X, min_nbins=50)

Estimate of the one-sigma error of the observable's mean.
Respects correlations between measurements through binning analysis.

Strategy: just take largest R value considering an upper limit for bin size (min_nbins)
"""
function binning_error_naive(X::AbstractVector{<:Number}, min_nbins::Int=50)
    # if not specified, choose binsize such that we have at least 50 full bins.
    binning_error(X, floor(Int, length(X)/min_nbins))
end
binning_error_naive(X::AbstractArray{<:Number}, min_nbins::Int=50) = begin nd = ndims(X); dropdims(mapslices(xi->binning_error_naive(xi, min_nbins=min_nbins), X, dims=nd), dims=nd) end
binning_error_naive(X::AbstractVector{<:AbstractArray}, min_nbins::Int=50) = binning_error_naive(cat(X..., dims=ndims(X[1])+1), min_nbins=min_nbins)