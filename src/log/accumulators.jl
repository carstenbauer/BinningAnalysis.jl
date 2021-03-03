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
    std_error(V::AbstractVarianceAccumulator)

Calculates the standard error of the mean for a given variance accumulator.
"""
function std_error(::AbstractVarianceAccumulator) end

std_error(V::AbstractVarianceAccumulator{T}) where {T <: Number} = sqrt(varN(V))
std_error(V::AbstractVarianceAccumulator{T}) where {T <: AbstractArray} = sqrt.(varN(V))


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


mutable struct FastVariance{T} <: AbstractVarianceAccumulator{T}
    x_sum::T
    x2_sum::T
    count::Int
end
FastVariance{T}() where T = FastVariance(zero(T), zero(T), zero(Int))
FastVariance() = FastVariance{Float64}()

@inline Base.isempty(V::FastVariance) = iszero(V.count)
function Base.empty!(V::FastVariance)
    V.x_sum, V.x2_sum = zero(V.x_sum), zero(V.x2_sum)
    V.count = zero(Int)
    V
end
Base.:(==)(a::FastVariance{T}, b::FastVariance{T}) where T = (
    (a.x_sum == b.x_sum)
    && (a.x2_sum == b.x2_sum)
    && (a.count == b.count)
)
Base.copy(V::FastVariance{T}) where T = FastVariance{T}(copy(V.x_sum), copy(V.x2_sum), copy(V.count))

mean(V::FastVariance) = isempty(V) ? zero(V.x_sum) : V.x_sum / V.count

function var(V::FastVariance{T}) where {T <: Real}
    n, X, X2 = V.count, V.x_sum, V.x2_sum
    X2 / (n - 1) - X^2 / (n*(n - 1))
end

function var(V::FastVariance{T}) where {T <: Complex}
    n, X, X2 = V.count, V.x_sum, V.x2_sum
    (real(X2) + imag(X2)) / (n - 1) - abs2(X) / (n*(n - 1))
end

function var(V::FastVariance{<: AbstractArray{T, D}}) where {D, T <: Real}
    n, X, X2 = V.count, V.x_sum, V.x2_sum
    @. X2 / (n - 1) - X^2 / (n*(n - 1))
end

function var(V::FastVariance{<: AbstractArray{T, D}}) where {D, T <: Complex}
    n, X, X2 = V.count, V.x_sum, V.x2_sum
    @. (real(X2) + imag(X2)) / (n - 1) - abs2(X) / (n*(n - 1))
end


function Base.push!(V::FastVariance{T}, value::T) where T
    V.x_sum += value
    V.x2_sum += _prod(value, value)
    V.count += 1
    return V
end

function Base.push!(V::FastVariance{T}, value::T) where T <: AbstractArray
    V.x_sum .+= value
    @. V.x2_sum += _prod(value, value)
    V.count += 1
    return V
end


mutable struct Variance{T} <: AbstractVarianceAccumulator{T}
    m1::T
    m2::T
    count::Int
end
Variance{T}() where T = Variance{T}(zero(T), zero(T), zero(Int))
Variance() = Variance{Float64}()

@inline Base.isempty(V::Variance) = iszero(V.count)
function Base.empty!(V::Variance)
    V.m1, V.m2 = zero(V.m1), zero(V.m2)
    V.count = zero(Int)
    V
end

Base.:(==)(a::Variance{T}, b::Variance{T}) where T = (
    (a.m1 == b.m1)
    && (a.m2 == b.m2)
    && (a.count == b.count)
)
Base.copy(V::Variance{T}) where T = Variance{T}(copy(V.m1), copy(V.m2), copy(V.count))

mean(V::Variance) = V.m1

var(V::Variance{T}) where {T <: Real} = V.m2 / (V.count - 1)
var(V::Variance{T}) where {T <: Complex} = (real(V.m2) + imag(V.m2)) / (V.count - 1)
var(V::Variance{<: AbstractArray{T, D}}) where {D, T <: Real} =
    @. V.m2 / (V.count - 1)
var(V::Variance{<: AbstractArray{T, D}}) where {D, T <: Complex} =
    @. (real(V.m2) + imag(V.m2)) / (V.count - 1)

function Base.push!(V::Variance{T}, value::S) where {S, T}
    δ = value - mean(V)
    V.count += 1
    V.m1 += δ / V.count
    V.m2 += _prod(δ, value - mean(V))
    return V
end


function Base.push!(V::Variance{T}, value::S) where {S, T <: AbstractArray}
    δ = value .- mean(V)
    V.count += 1
    V.m1 .+= δ / V.count
    V.m2 .+= _prod(δ, value .- mean(V))
    return V
end
