# This is an attempt of implementing a time series buffer.
# Basically we want to augment an array `ts_buffer` with some
# preallocation logic.
# 
# For "full binning", we operate on the full time series anyway,
# hence we should provide full binning tools for any one-dimensional
# (maybe <:AbstractArray) container.
#
# Note that TSBuffer is not quite <: AbstractArray yet,
# see https://docs.julialang.org/en/latest/manual/interfaces/#man-interface-array-1
struct TSBuffer{T}
    # time series buffer
    ts_buffer::Vector{T}

    # number of elements currently stored in ts_buffer
    count::Base.RefValue{Int64}

    # enlarge ts_buffer by `alloc` elements on overflow
    alloc::Int64
end




"""
    TSBuffer([::Type{T}; sizehint::Int])

Creates a `TSBuffer` which can handle values of type `T`.

The default is `T = Float64`.
"""
TSBuffer(::Type{T} = Float64; kw...) where T = TSBuffer(zero(T); kw...)


"""
    TSBuffer([zero_element::T; sizehint::Int=1000])

Creates a new `TSBuffer` which can take values of type `T`. The type and size
is inherited by the given `zero_element`.

Values can be added using `push!` and `append!`.
"""
function TSBuffer(_zero::T;
        sizehint::Integer = 1000,
        alloc::Integer = ceil(Int, sizehint*0.1) # 10% of sizehint
        ) where {T}

    # check keyword args
    sizehint <= 0 && throw(ArgumentError("`sizehint` must be finite and positive."))

    TSBuffer{T}(
        Vector{T}(undef, sizehint),
        Ref(0),
        alloc
    )
end






@inline Base.length(B::TSBuffer) = B.count[]
@inline Base.lastindex(B::TSBuffer) = length(B)
@inline Base.eltype(B::TSBuffer{T}) where T = T
@inline Base.isempty(B::TSBuffer) = length(B) == 0

@inline _buffer_length(B::TSBuffer) = length(B.ts_buffer)
@inline _isfull(B::TSBuffer) = length(B) == _buffer_length(B)
@inline _resize!(B::TSBuffer, ntimes::Integer=1) = resize!(B.ts_buffer, _buffer_length(B) + ntimes * B.alloc)


function Base.sizehint!(B::TSBuffer, n::Integer)
    n >= length(TSBuffer) || (return nothing) # we'd loose elements, so ignore sizehint
    resize!(B.ts_buffer, n)
    nothing
end

function Base.resize!(B::TSBuffer, n::Integer)
    resize!(B.ts_buffer, n)
    if length(B) > n
        B.count[] = n
    end
    nothing
end

function Base.empty!(B::TSBuffer)
    B.count[] = 0
    # should we resize (i.e. shorten) ts_buffer here?
    nothing
end





@inline Base.checkbounds(::Type{Bool}, B::TSBuffer, I::AbstractArray) = all(i -> i in Base.OneTo(length(B)), I)
@inline Base.checkbounds(::Type{Bool}, B::TSBuffer, i::Integer) = i in Base.OneTo(length(B))
@inline function Base.checkbounds(B::TSBuffer, I)
    checkbounds(Bool, B, I) || throw(BoundsError(B, I))
    nothing
end




function Base.getindex(B::TSBuffer, I::Union{Integer, AbstractArray})
    checkbounds(B, I)
    @inbounds getindex(B.ts_buffer, I)
end


function Base.view(B::TSBuffer, I::Union{Integer, AbstractArray})
    checkbounds(B, I)
    @inbounds view(B.ts_buffer, I)
end



timeseries(B::TSBuffer) = B[1:end]
ts(B::TSBuffer) = timeseries(B)








# cosmetics
function _print_header(io::IO, B::TSBuffer{T}) where T
    print(io, "TSBuffer{$(T)}")
    nothing
end

function _println_body(io::IO, B::TSBuffer{T}) where T
    n = length(B)
    println(io)
    print(io, "| Count: ", n)
    # if n > 0 && ndims(B) == 0
    #     print(io, "\n| Mean: ", round.(mean(B), digits=5))
    #     print(io, "\n| StdError: ", round.(std_error(B), digits=5))
    # end
    nothing
end

# short version (shows up in arrays etc.)
Base.show(io::IO, B::TSBuffer{T}) where T = print(io, "TSBuffer{$(T)}()")
# verbose version (shows up in the REPL)
Base.show(io::IO, m::MIME"text/plain", B::TSBuffer) = (_print_header(io, B); _println_body(io, B))












@inline function _push_to_buffer!(B::TSBuffer, v)
    # resize if necessary to avoid overflow
    _isfull(B) && _resize!(B)

    # add element to buffer
    @inbounds B.ts_buffer[B.count[] + 1] = v
    B.count[] += 1
    nothing
end


@inline function _append_to_buffer!(B::TSBuffer, vs)
    n = length(vs)
    l = _buffer_length(B)
    c = length(B)

    if c + n > l # overflow
        extby = (c + n) - l
        ntimes = ceil(Int, extby / B.alloc)
        _resize!(B, ntimes)
    end

    # add elements to buffer (TODO: compare speed to for loop)
    @inbounds B.ts_buffer[c+1:c+n] = vs
    B.count[] += n
    nothing
end





"""
    append!(TSBuffer, values)

Adds an array of values to the binner by `push!`ing each element.
"""
function Base.append!(B::TSBuffer{T}, values::AbstractArray{T}) where T
    _append_to_buffer!(B, values)
    nothing
end



"""
    push!(TSBuffer, value)

Pushes a new value into the Binning Analysis.
"""
function Base.push!(B::TSBuffer{T}, value::T) where T
    _push_to_buffer!(B, value)
    nothing
end