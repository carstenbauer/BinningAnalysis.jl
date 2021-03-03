# This is a Node in the Binning Analysis tree that averages two values. There
# is one of these for each binning level. When two values should be
# compressed, this is done immediately, so that only one value needs to be saved.
# switch indicates whether value should be written to or averaging should happen.
mutable struct Compressor{T}
    value::T
    switch::Bool
end


struct LogBinner{T, N, V <: AbstractVarianceAccumulator{T}}
    # list of Compressors, one per level
    compressors::NTuple{N, Compressor{T}}

    accumulators::NTuple{N, V}

    function LogBinner{T, N}(
            compressors::NTuple{N, Compressor{T}},
            accumulators::NTuple{N, V}
        ) where {T, N, V <: AbstractVarianceAccumulator{T}}
        new{T, N, V}(compressors, accumulators)
    end
end




# Overload some basic Base functions
Base.eltype(B::LogBinner{T,N}) where {T,N} = T
Base.length(B::LogBinner) = B.accumulators[1].count
Base.ndims(B::LogBinner{T,N}) where {T,N} = ndims(eltype(B))
Base.isempty(B::LogBinner) = length(B) == 0
Base.:(==)(a::T, b::T) where {T <: Compressor} = (a.value == b.value) && (a.switch == b.switch)
Base.:(!=)(a::T, b::T) where {T <: Compressor} = !(a == b)

function Base.:(==)(a::LogBinner{T, N}, b::LogBinner{T, M}) where {T, N, M}
    # Switch order so that we can deal with just N ≤ M here
    (N > M) && (return b == a)

    # Does every level match?
    for i in 1:N
        (a.compressors[i] == b.compressors[i]) || return false
        (a.accumulators[i] == b.accumulators[i]) || return false
    end

    # Are extra levels empty?
    for i in N+1:M
        isempty(b.accumulators[i]) || return false
    end

    return true
end
Base.:(!=)(a::LogBinner{T, N}, b::LogBinner{T, M}) where {T, N, M} = !(a == b)



function _print_header(io::IO, B::LogBinner{T,N}) where {T,N}
    print(io, "LogBinner{$(T),$(N)}")
    nothing
end

function _println_body(io::IO, B::LogBinner{T,N}) where {T,N}
    n = length(B)
    println(io)
    print(io, "| Count: ", n)
    if n > 0 && ndims(B) == 0
        print(io, "\n| Mean: ", round.(mean(B), digits=5))
        print(io, "\n| StdError: ", round.(std_error(B), digits=5))
    end
    nothing
end

# short version (shows up in arrays etc.)
Base.show(io::IO, B::LogBinner{T,N}) where {T,N} = print(io, "LogBinner{$(T),$(N)}()")
# verbose version (shows up in the REPL)
Base.show(io::IO, m::MIME"text/plain", B::LogBinner) = (_print_header(io, B); _println_body(io, B))





"""
    empty!(B::LogBinner)

Clear the binner, i.e. reset it to its inital state.
"""
function Base.empty!(B::LogBinner)
    !isempty(B) || return

    # get zero
    z = zero(eltype(eltype(B)))

    # reset accumulators
    for acc in B.accumulators
        empty!(acc)
    end

    # reset compressors
    for c in B.compressors
        if ndims(B) == 0 # compile time
            c.value = z
        else
            fill!(c.value, z)
        end
        c.switch = false
    end

    nothing
end



_nlvls2capacity(N::Int) = 2^N - 1
_capacity2nlvls(capacity::Int) = ceil(Int, log(2, capacity + 1))

"""
    capacity(B)

Capacity of the binner, i.e. how many values can be handled before overflowing.
"""
capacity(B::LogBinner{T,N}) where {T,N} = _nlvls2capacity(N)
nlevels(B::LogBinner{T,N}) where {T,N} = N



"""
    LogBinner([::Type{T}; capacity::Int])

Creates a `LogBinner` which can handle (at least) `capacity` many values of type `T`.

The default is `T = Float64` and `capacity = 2^32-1 ≈ 4e9`.
"""
LogBinner(::Type{T} = Float64; kw...) where T = LogBinner(zero(T); kw...)

# TODO
# Currently, the Binning Analysis requires a "zero" to initialize x_sum and
# x2_sum. It's not necessary for the Compressors.
# Since Arrays do not have a static size, we cannot generate a fitting zero
# automatically. So currently we let the user supply it.
# Other possibilities:
# > generate x_sum, x2_sum with #undef
# bad: requires many checks (if isassigned ... else ...)
# > force StaticArrays (and/or tuples)
# good: faster for small arrays
# bad: requires user to use another package
# > initialize is first push!
# bad: also requires frequent checks (if first push ... else ...)
# ...?
"""
    LogBinner(zero_element::T[; capacity::Int])

Creates a new `LogBinner` which can take (at least) `capacity` many values of type `T`. The type
and the size are inherited from the given `zero_element`, which must exclusively contain zeros.

Values can be added using `push!` and `append!`.

---

    LogBinner(timeseries::AbstractVector{T})

Creates a new `LogBinner` and adds all elements from the given timeseries.
"""
function LogBinner(x::T;
        capacity::Int64 = _nlvls2capacity(32),
        accumulator::Type{<:AbstractVarianceAccumulator} = Variance
    ) where {T <: Union{Number, AbstractArray}}

    # check keyword args
    capacity <= 0 && throw(ArgumentError("`capacity` must be finite and positive."))

    # got_timeseries = didn't receive a zero && is a vector
    got_timeseries = count(!iszero, x) > 0 && ndims(T) == 1

    if got_timeseries
        # x = time series
        N = _capacity2nlvls(length(x))
        S = _sum_type_heuristic(eltype(T), ndims(x[1]))
        el = zero(x[1])
    else
        # x = zero_element
        N = _capacity2nlvls(capacity)
        S = _sum_type_heuristic(T, ndims(x))
        el = x
    end

    B = LogBinner{S, N}(
        tuple([Compressor{S}(copy(el), false) for i in 1:N]...),
        tuple([accumulator{S}(zero(el)) for _ in 1:N]...)
    )

    got_timeseries && append!(B, x)

    return B
end

function _sum_type_heuristic(::Type{T}, elndims::Integer) where T
    # heuristic to set sum type (#2)
    S = if eltype(T)<:Real
        elndims > 0 ? Array{Float64, elndims} : Float64
    else
        elndims > 0 ? Array{ComplexF64, elndims} : ComplexF64
    end
    return S
end

"""
    LogBinner(B::LogBinner[; capacity::Int])

Creates a new `LogBinner` from an existing LogBinner, copying the data inside.
The new LogBinner may be larger or smaller than the given one.
"""
function LogBinner(B::LogBinner{S, M}; capacity::Int64 = _nlvls2capacity(32)) where {S, M}
    N = _capacity2nlvls(capacity)
    B.accumulators[min(M, N)].count > 1 && throw(OverflowError(
        "The new LogBinner is too small to reconstruct the given LogBinner. " *
        "New capacity = $capacity   Old capacity = $(B.accumulators[1].count)"
    ))
    el = zero(mean(B.accumulators[1]))
    V = empty!(copy(B.accumulators[1]))

    LogBinner{S, N}(
        tuple([i > M ? Compressor{S}(copy(el), false) : deepcopy(B.compressors[i]) for i in 1:N]...),
        tuple([i > M ? copy(V) : copy(B.accumulators[i]) for i in 1:N]...)
    )
end



# TODO typing?
"""
    append!(LogBinner, values)

Adds an array of values to the binner by `push!`ing each element.
"""
function Base.append!(B::LogBinner, values::AbstractArray)
    @inbounds for i in eachindex(values)
        _push!(B, 1, values[i])
    end
    nothing
end


"""
    push!(LogBinner, value)

Pushes a new value into the Binning Analysis.
"""
function Base.push!(B::LogBinner{T,N}, value::S) where {N, T, S}
    ndims(T) == ndims(S) || throw(DimensionMismatch("Expected $(ndims(T)) dimensions but got $(ndims(S))."))

    _push!(B, 1, value)
end

# recursion, back-end function
@inline function _push!(B::LogBinner{T,N}, lvl::Int64, value::S) where {N, T <: Number, S}
    C = B.compressors[lvl]

    # any value propagating through this function is new to lvl. Therefore we
    # add it to the sums. Note that values pushed to the output arrays are not
    # added here until the array drops to the next level. (New compressors are
    # added)
    Base.push!(B.accumulators[lvl], value)

    if !C.switch
        # Compressor has space -> save value
        C.value = value
        C.switch = true
        return nothing
    else
        # Do averaging
        if lvl == N
            # No more propagation possible -> throw error
            throw(OverflowError("The Binning Analysis has exceeded its maximum capacity."))
        else
            # propagate to next lvl
            C.switch = false
            _push!(B, lvl+1, 0.5 * (C.value + value))
            return nothing
        end
    end
    return nothing
end

function _push!(
        B::LogBinner{T,N},
        lvl::Int64,
        value::S
    ) where {N, T <: AbstractArray, S}

    C = B.compressors[lvl]
    Base.push!(B.accumulators[lvl], value)

    if !C.switch
        C.value .= value
        C.switch = true
        return nothing
    else
        if lvl == N
            throw(OverflowError("The Binning Analysis has exceeded its maximum capacity."))
        else
            C.switch = false
            @. C.value = 0.5 * (C.value + value)
            _push!(B, lvl+1, C.value)
            return nothing
        end
    end
    return nothing
end
