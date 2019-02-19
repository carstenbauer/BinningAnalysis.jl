"""
**Jackknife** errors for (non-linear) functions of uncertain data, i.e.
`g(<a>, <b>, ...)` where `<a>`, `<b>` are the means of data sets `a` and `b`.
Use `jackknife_full(g, a, b, ...)` to get the jackknife estimate, bias and
error or `jackknife(g, a, b, ...)` to get just the estimate and error.

See: [`jackknife_full`](@ref), [`jackknife`](@ref)
"""
module Jackknife



import Statistics: mean, var

export jackknife_full, jackknife


"""
    jackknife_full(g::Function, a[, b, ...])

Returns the jackknife estimate, bias and error for a given function
`g(<a>, <b>, ...)` acting on the means of the samples `a`, `b`, etc.

Example:
    # Assuming some sample xs is given
    # variance of sample xs
    g(x2, x) = x2 - x^2
    estimate, bias, error = jackknife_full(g, xs.^2, xs)

See also: [`estimate`](@ref), [`bias`](@ref), [`std_error`](@ref)
"""
function jackknife_full(g::Function, samples::AbstractVector{<:Number}...)
    reduced_results = leaveoneout(g, samples...)
    return estimate(g, samples...; reduced_results = reduced_results),
           bias(g, samples...; reduced_results = reduced_results),
           _std_error(reduced_results)
end
function jackknife_full(g::Function, samples::AbstractArray{<:Number})
    jackknife_full(g, [samples[:, i] for i in 1:size(samples, 2)]...)
end

"""
    jackknife(g::Function, a[, b, ...])

Returns the jackknife estimate and error for a given function `g(<a>, <b>, ...)`
acting on the means of the samples `a`, `b`, etc.

Example:
    # Assuming some sample xs is given
    # variance of sample xs
    g(x2, x) = x2 - x^2
    error = jackknife(g, xs.^2, xs)

See also: [`estimate`](@ref), [`std_error`](@ref)
"""
function jackknife(g::Function, samples::AbstractVector{<:Number}...)
    reduced_results = leaveoneout(g, samples...)
    return estimate(g, samples...; reduced_results = reduced_results),
           _std_error(reduced_results)
end
function jackknife(g::Function, samples::AbstractArray{<:Number})
    jackknife(g, [samples[:, i] for i in 1:size(samples, 2)]...)
end


"""
    leaveoneout(g::Function, samples::AbstractVector{<:Number}...)

Generates N sub-samples for each sample in `samples`, where one value is left
out and applies it mean to the function `g`. The result is used for a
1-jackknife.
"""
function leaveoneout(g::Function, samples::AbstractVector{<:Number}...)
    @assert(
        length(samples[1]) > 1,
        "The sample must have multiple values! (length(samples[1]) > 1)"
    )

    full_sample_sums = map(sum, samples)
    invN1 = 1 / (length(samples[1]) - 1)
    red_sample_means = map(samples, full_sample_sums) do sample, full_sample_sum
        [(full_sample_sum - sample[i]) * invN1 for i in eachindex(sample)]
    end
    g.(red_sample_means...)
end
function leaveoneout(g::Function, samples::AbstractArray{<:Number})
    leaveoneout(g, [samples[:, i] for i in 1:size(samples, 2)]...)
end


"""
    var(g::Function, a[, b, ...])

Returns the jackknife variance for a given function `g(<a>, <b>, ...)` acting on
the means of the samples `a`, `b`, etc.
"""
function var(g::Function, samples::AbstractVector{<:Number}...)
    _var(leaveoneout(g, samples...))
end
function var(g::Function, samples::AbstractArray{<:Number})
    var(g, [samples[:, i] for i in 1:size(samples, 2)]...)
end
_var(gis::AbstractVector{<:Complex}) = _var(real(gis)) + _var(imag(gis))
function _var(reduced_results::AbstractVector{<:Real})
    # With N values each sample, N subsamples are created and N reduced_results
    # are calculated.
    N = length(reduced_results)
    # Eq. (3.35) in QMC Methods book
    return var(reduced_results) * (N - 1)^2 / N
end


"""
    std_error(g::Function, a[, b, ...])

Returns the jackknife error for a given function `g(<a>, <b>, ...)` acting on
the means of the samples `a`, `b`, etc.
"""
function std_error(g::Function, samples::AbstractVector{<:Number}...)
    _std_error(leaveoneout(g, samples...))
end
function std_error(g::Function, samples::AbstractArray{<:Number})
    std_error(g, [samples[:, i] for i in 1:size(samples, 2)]...)
end
function _std_error(gis::AbstractVector{<:Complex})
    sqrt(_std_error(real(gis))^2 + _std_error(imag(gis))^2)
end
_std_error(gis::AbstractVector{<:Real}) = sqrt(_var(gis))


"""
    bias(g::Function, a[, b, ...])

Returns the jackknife bias for a given function `g(<a>, <b>, ...)` acting on
the means of the samples `a`, `b`, etc.
"""
function bias(
        g::Function,
        samples::AbstractVector{<:Number}...;
        reduced_results::AbstractVector{<:Number} = leaveoneout(g, samples...)
    )
    # Basically Eq. (3.33)
    (length(reduced_results) - 1) * (
        mean(reduced_results) - g(map(mean, samples)...)
    )
end
function bias(g::Function, samples::AbstractArray{<:Number})
    bias(g, [samples[:, i] for i in 1:size(samples, 2)]...)
end


"""
    estimate(g::Function, a[, b, ...])

Returns the (bias corrected) jackknife estimate for a given function
`g(<a>, <b>, ...)` acting on the means of the samples `a`, `b`, etc. If
`leaveoneout` has already been calculated it can be supplied via the keyword
argument `reduced_results`.
"""
function estimate(
        g::Function,
        samples::AbstractVector{<:Number}...;
        reduced_results::AbstractVector{<:Number} = leaveoneout(g, samples...)
    )
    # Eq. (3.34) in QMC Methods book
    n = length(reduced_results)
    return n * g(map(mean, samples)...) - (n - 1) * mean(reduced_results)
end
function estimate(g::Function, samples::AbstractArray{<:Number})
    estimate(g, [samples[:, i] for i in 1:size(samples, 2)]...)
end



end # module


# TODO: Prebinning + Jackknife
