abstract type AbstractVarianceAccumulator{T} end


"""
    isempty(V::AbstractVarianceAccumulator)

Returns true if the given variance accumulator is empty.
"""
function Base.isempty(::AbstractVarianceAccumulator) end


"""
    empty!(V::AbstractVarianceAccumulator)

Clear the given variance accumulator.
"""
function Base.empty!(::AbstractVarianceAccumulator) end

Base.:(==)(a::AbstractVarianceAccumulator, b::AbstractVarianceAccumulator) = false
Base.isapprox(a::AbstractVarianceAccumulator, b::AbstractVarianceAccumulator; kwargs...) = false


"""
    mean(V::AbstractVarianceAccumulator)

Calculates the mean of a given variance accumulator.
"""
function mean(V::AbstractVarianceAccumulator) end


"""
    varN(V::AbstractVarianceAccumulator)

Calculates the variance/N of a given variance accumulator.
"""
varN(V::AbstractVarianceAccumulator{T}) where T = var(V) / V.count


"""
    var(V::AbstractVarianceAccumulator)

Calculates the variance of a given variance accumulator.
"""
function var(::AbstractVarianceAccumulator) end


"""
    push!(V::AbstractVarianceAccumulator{T}, value::T)

Pushes a new value into the variance accumulator.
"""
function Base.push!(::AbstractVarianceAccumulator{T}, value::T) where T end



@inline _prod(x::T, y::T) where T = x * y
@inline _prod(x::Complex, y::Complex) = Complex(real(x) * real(y), imag(x) * imag(y))
@inline _prod(x::AbstractArray, y::AbstractArray) = _prod.(x, y)



"""
    FastVariance <: AbstractVarianceAccumulator

The `FastVariance` accumulator keeps track of `∑x` and `∑x²` where `x` refers to
the pushed values. It is significantly faster than the `Variance` accumulator,
but can become unstable for some inputs. (For example when the mean is much 
large than the variance of the pushed values.)

See also: [`AbstractVarianceAccumulator`](@ref), [`Variance`](@ref)
"""
mutable struct FastVariance{T} <: AbstractVarianceAccumulator{T}
    x_sum::T
    x2_sum::T
    count::Int
end
function FastVariance{T}(T0) where T 
    FastVariance{T}(deepcopy(T0), deepcopy(T0), zero(Int))
end
FastVariance{T}() where T = FastVariance(zero(T), zero(T), zero(Int))
FastVariance() = FastVariance{Float64}()

@inline Base.isempty(V::FastVariance) = iszero(V.count)
function Base.empty!(V::FastVariance)
    V.x_sum, V.x2_sum = zero(V.x_sum), zero(V.x2_sum)
    V.count = zero(Int)
    return V
end
function Base.isapprox(a::FastVariance{T}, b::FastVariance{T}; kwargs...) where T
    return isapprox(a.x_sum, b.x_sum; kwargs...) && 
           isapprox(a.x2_sum, b.x2_sum; kwargs...) && 
           isapprox(a.count, b.count; kwargs...)
end
function Base.:(==)(a::FastVariance{T}, b::FastVariance{T}) where T
    return (a.x_sum == b.x_sum) && (a.x2_sum == b.x2_sum) && (a.count == b.count)
end
function Base.copy(V::FastVariance{T}) where T
    return FastVariance{T}(copy(V.x_sum), copy(V.x2_sum), copy(V.count))
end

mean(V::FastVariance) = isempty(V) ? zero(V.x_sum) : V.x_sum / V.count

function var(V::FastVariance{T}) where {T <: Real}
    n, X, X2 = V.count, V.x_sum, V.x2_sum
    return X2 / (n - 1) - X^2 / (n*(n - 1))
end

function var(V::FastVariance{T}) where {T <: Complex}
    n, X, X2 = V.count, V.x_sum, V.x2_sum
    return (real(X2) + imag(X2)) / (n - 1) - abs2(X) / (n*(n - 1))
end

function var(V::FastVariance{<: AbstractArray{T, D}}) where {D, T <: Real}
    n, X, X2 = V.count, V.x_sum, V.x2_sum
    return @. X2 / (n - 1) - X^2 / (n*(n - 1))
end

function var(V::FastVariance{<: AbstractArray{T, D}}) where {D, T <: Complex}
    n, X, X2 = V.count, V.x_sum, V.x2_sum
    return @. (real(X2) + imag(X2)) / (n - 1) - abs2(X) / (n*(n - 1))
end


function _push!(V::FastVariance{T}, value::T) where T
    V.x_sum += value
    V.x2_sum += _prod(value, value)
    V.count += 1
    return V
end

function _push!(V::FastVariance{T}, value::T) where T <: AbstractArray
    V.x_sum .+= value
    @. V.x2_sum += _prod(value, value)
    V.count += 1
    return V
end



"""
    Variance <: AbstractVarianceAccumulator

The `Variance` accumulator uses Welfords algorithm to keep track of pushed 
values and to compute variances and errors. It is less error prone than 
`FastVariance` but also slower.

See also: [`AbstractVarianceAccumulator`](@ref), [`FastVariance`](@ref)
"""
mutable struct Variance{T} <: AbstractVarianceAccumulator{T}
    δ::T
    m1::T
    m2::T
    count::Int
end
function Variance{T}(T0) where {T}
    Variance{T}(deepcopy(T0), deepcopy(T0), deepcopy(T0), zero(Int))
end
Variance{T}() where T = Variance{T}(zero(T), zero(T), zero(T), zero(Int))
Variance() = Variance{Float64}()

@inline Base.isempty(V::Variance) = iszero(V.count)
function Base.empty!(V::Variance)
    V.m1, V.m2 = zero(V.m1), zero(V.m2)
    V.count = zero(Int)
    return V
end

function Base.isapprox(a::Variance{T}, b::Variance{T}; kwargs...) where T
    return isapprox(a.m1, b.m1; kwargs...) && 
           isapprox(a.m2, b.m2; kwargs...) && 
           isapprox(a.count, b.count; kwargs...)
end
function Base.:(==)(a::Variance{T}, b::Variance{T}) where T
    return (a.m1 == b.m1) && (a.m2 == b.m2) && (a.count == b.count)
end
function Base.copy(V::Variance{T}) where T
    return Variance{T}(copy(V.δ), copy(V.m1), copy(V.m2), copy(V.count))
end

mean(V::Variance) = V.m1

var(V::Variance{T}) where {T <: Real} = V.m2 / (V.count - 1)
var(V::Variance{T}) where {T <: Complex} = (real(V.m2) + imag(V.m2)) / (V.count - 1)
var(V::Variance{<: AbstractArray{T, D}}) where {D, T <: Real} =
    @. V.m2 / (V.count - 1)
var(V::Variance{<: AbstractArray{T, D}}) where {D, T <: Complex} =
    @. (real(V.m2) + imag(V.m2)) / (V.count - 1)

function _push!(V::Variance{T}, value::S) where {S, T}
    δ = value - mean(V)
    V.count += 1
    V.m1 += δ / V.count
    V.m2 += _prod(δ, value - mean(V))
    return V
end


function _push!(V::Variance{T}, value::S) where {S, T <: AbstractArray}
    @. V.δ = value - V.m1
    V.count += 1
    @. V.m1 += V.δ / V.count
    @. V.m2 += _prod(V.δ, value - V.m1)
    return V
end
