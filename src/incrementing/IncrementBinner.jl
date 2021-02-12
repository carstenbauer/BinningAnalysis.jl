mutable struct IncrementBinner{T}
    keep::Int64         # number of values per block
    compression::Int64  # current N-mean
    stage::Int64        # stage * keep == current max length
    
    # for local averages
    sum::T
    count::Int64

    output::Vector{T}
end

# Overload some basic Base functions
Base.eltype(B::IncrementBinner{T}) where {T} = T
Base.length(B::IncrementBinner) = length(B.output)
Base.ndims(B::IncrementBinner{T}) where {T} = ndims(eltype(B))
Base.isempty(B::IncrementBinner) = length(B) == 0

Base.show(io::IO, B::IncrementBinner{T}) where {T} = print(io, "IncrementBinner{$T}()")

"""
    IncrementBinner([::Type{T}; blocksize = 64])
    IncrementBinner(zero_element::T[; blocksize = 64])

Creates an `IncrementBinner` from a `zero_element` numeric type `T`.

Values pushed to an `IncrementBinner` are averaged in stages. For the first 
`blocksize` values pushed no averaging happens. After that `2blocksize` elements
are averaged 2 at a time, then `4blocksize` element 4 at a time, etc. This means 
values pushed become progressively more compressed and smoothed.
"""
IncrementBinner(::Type{T} = Float64; kw...) where {T} = IncrementBinner(zero(T); kw...)
function IncrementBinner(x::T; blocksize = 64) where {T <: Union{Number, AbstractArray}}
    # check keyword args
    blocksize <= 0 && throw(ArgumentError("`blocksize` must be finite and positive."))

    # got_timeseries = didn't receive a zero && is a vector
    got_timeseries = count(!iszero, x) > 0 && ndims(T) == 1

    if got_timeseries
        # x = time series
        S = _sum_type_heuristic(eltype(T), ndims(x[1])) # from log
        el = zero(x[1])
    else
        # x = zero_element
        S = _sum_type_heuristic(T, ndims(x))
        el = x
    end

    B = IncrementBinner(blocksize, 1, 1, el, 0, S[])

    got_timeseries && append!(B, x)

    return B
end

function Base.push!(b::IncrementBinner{T}, x::T) where {T}
    if b.count >= b.compression
        push!(b.output, b.sum / b.compression)
        b.count = 0
        b.sum = x
    else
        b.count += 1
        b.sum += x
    end
    if length(b.output) == b.stage * b.keep
        b.compression *= 2
        b.stage += 1
    end
    nothing
end

function Base.push!(b::IncrementBinner{T}, x::T) where {T <: AbstractArray}
    if b.count >= b.compression
        push!(b.output, b.sum / b.compression)
        b.count = 0
        b.sum .= x
    else
        b.count += 1
        b.sum .+= x
    end
    if length(b.output) == b.stage * b.keep
        b.compression *= 2
        b.stage += 1
    end
    nothing
end

function Base.append!(b::IncrementBinner{T}, v::Vector{T}) where {T}
    for x in v
        push!(b, x)
    end
    nothing
end

Base.values(B::IncrementBinner) = B.output

function indices(B::IncrementBinner)
    out = zeros(length(B.output))
    step = 1
    out[1] = 1
    for i in 2:length(B.output)
        if i % B.keep == 0
            out[i] = out[i-1] + 1.5step 
            step *= 2
        else
            out[i] = out[i-1] + step
        end
    end
    out
end