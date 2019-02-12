struct FullBinner{T}
    # time series buffer
    ts_buffer::Vector{T}

    # number of elements currently stored in ts_buffer
    count::Base.RefValue{Int64}

    # enlarge ts_buffer by `alloc` elements on overflow
    alloc::Int64
end




"""
    FullBinner([::Type{T}; sizehint::Int])

Creates a `FullBinner` which can handle values of type `T`.

The default is `T = Float64`.
"""
FullBinner(::Type{T} = Float64; kw...) where T = FullBinner(zero(T); kw...)


"""
    FullBinner([zero_element::T; sizehint::Int=1000])

Creates a new `FullBinner` which can take values of type `T`. The type and size
is inherited by the given `zero_element`.

Values can be added using `push!` and `append!`.
"""
function FullBinner(_zero::T;
        sizehint::Integer = 1000,
        alloc::Integer = ceil(Int, sizehint*0.1) # 10% of sizehint
        ) where {T}

    # check keyword args
    sizehint <= 0 && throw(ArgumentError("`sizehint` must be finite and positive."))

    FullBinner{T}(
        Vector{T}(undef, sizehint),
        Ref(0),
        alloc
    )
end






@inline Base.length(F::FullBinner) = F.count[]
@inline Base.eltype(F::FullBinner{T}) where T = T
@inline Base.isempty(F::FullBinner) = length(F) == 0

@inline _buffer_length(F::FullBinner) = length(F.ts_buffer)
@inline _isfull(F::FullBinner) = length(F) == _buffer_length(F)
@inline _resize!(F::FullBinner, ntimes::Integer=1) = resize!(F.ts_buffer, _buffer_length(F) + ntimes * F.alloc)


function Base.sizehint!(F::FullBinner, n::Integer)
    n >= length(FullBinner) || (return nothing) # we'd loose elements, so ignore sizehint
    resize!(F.ts_buffer, n)
    nothing
end

function Base.resize!(F::FullBinner, n::Integer)
    resize!(F.ts_buffer, n)
    if length(F) > n
        F.count[] = n
    end
    nothing
end

function Base.empty!(F::FullBinner)
    F.count[] = 0
    # should we resize (i.e. shorten) ts_buffer here?
    nothing
end





@inline Base.checkbounds(::Type{Bool}, F::FullBinner, I::AbstractArray) = all(i -> i in Base.OneTo(length(F)), I)
@inline Base.checkbounds(::Type{Bool}, F::FullBinner, i::Integer) = i in Base.OneTo(length(F))
@inline function Base.checkbounds(F::FullBinner, I)
    checkbounds(Bool, F, I) || throw(BoundsError(F, I))
    nothing
end


# function Base.getindex(F::FullBinner, idx::Integer)
#     idx > length(F) && throw()

#     return getindex
# end




timeseries(F::FullBinner) = @inbounds F.ts_buffer[1:length(F)]
ts(F::FullBinner) = timeseries(F)








# cosmetics
function _print_header(io::IO, F::FullBinner{T}) where T
    print(io, "FullBinner{$(T)}")
    nothing
end

function _println_body(io::IO, F::FullBinner{T}) where T
    n = length(F)
    println(io)
    print(io, "| Count: ", n)
    # if n > 0 && ndims(F) == 0
    #     print(io, "\n| Mean: ", round.(mean(F), digits=5))
    #     print(io, "\n| StdError: ", round.(std_error(F), digits=5))
    # end
    nothing
end

# short version (shows up in arrays etc.)
Base.show(io::IO, F::FullBinner{T}) where T = print(io, "FullBinner{$(T)}()")
# verbose version (shows up in the REPL)
Base.show(io::IO, m::MIME"text/plain", F::FullBinner) = (_print_header(io, F); _println_body(io, F))












@inline function _push_to_buffer!(F::FullBinner, v)
    # resize if necessary to avoid overflow
    _isfull(F) && _resize!(F)

    # add element to buffer
    @inbounds F.ts_buffer[F.count[] + 1] = v
    F.count[] += 1
    nothing
end


@inline function _append_to_buffer!(F::FullBinner, vs)
    n = length(vs)
    l = _buffer_length(F)
    c = length(F)

    if c + n > l # overflow
        extby = (c + n) - l
        ntimes = ceil(Int, extby / F.alloc)
        _resize!(F, ntimes)
    end

    # add elements to buffer (TODO: compare speed to for loop)
    @inbounds F.ts_buffer[c+1:c+n] = vs
    F.count[] += n
    nothing
end





"""
    append!(FullBinner, values)

Adds an array of values to the binner by `push!`ing each element.
"""
function append!(F::FullBinner{T}, values::AbstractArray{T}) where T
    _append_to_buffer!(F, values)
    nothing
end



"""
    push!(FullBinner, value)

Pushes a new value into the Binning Analysis.
"""
function push!(F::FullBinner{T}, value::T) where T
    _push_to_buffer!(F, value)
    nothing
end