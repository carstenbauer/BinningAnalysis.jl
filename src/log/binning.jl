# This is a Node in the Binning Analysis tree that averages two values. There
# is one of these for each binning level. When two values should be
# compressed, this is done immediately, so that only one value needs to be saved.
# switch indicates whether value should be written to or averaging should happen.
mutable struct Compressor{T}
    value::T
    switch::Bool
end


struct LogBinner{N, T}
    # list of Compressors, one per level
    compressors::NTuple{N, Compressor{T}}

    # sum(x) for all values on a given lvl
    x_sum::Vector{T}
    # sum(x.^2) for all values on a given lvl
    x2_sum::Vector{T}
    # number of values that are summed on a given lvl
    count::Vector{Int64}
end



# Overload some basic Base functions
Base.eltype(B::LogBinner{N,T}) where {N,T} = T
Base.length(B::LogBinner) = B.count[1]
Base.ndims(B::LogBinner{N,T}) where {N,T} = ndims(eltype(B))
Base.isempty(B::LogBinner) = length(B) == 0





function _print_header(io::IO, B::LogBinner{N,T}) where {N, T}
    print(io, "LogBinner{$(N),$(T)}")
    nothing
end

function _println_body(io::IO, B::LogBinner{N,T}) where {N, T}
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
Base.show(io::IO, B::LogBinner{N,T}) where {N, T} = print(io, "LogBinner{$(N),$(T)}()")
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

    # reset x_sum and x2_sum to all zeros
    @inbounds for i in eachindex(B.x_sum)
        if ndims(B) == 0 # compile time
            # numbers
            B.x_sum[i] = z
            B.x2_sum[i] = z
        else 
            # arrays
            fill!(B.x_sum[i], z)
            fill!(B.x2_sum[i], z)
        end
    end

    # reset counts
    fill!(B.count, 0)

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
capacity(B::LogBinner{N, T}) where {N,T} = _nlvls2capacity(N)
nlevels(B::LogBinner{N, T}) where {N,T} = N



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
        capacity::Int64 = _nlvls2capacity(32)
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

    B = LogBinner{N, S}(
        tuple([Compressor{S}(copy(el), false) for i in 1:N]...),
        [copy(el) for _ in 1:N],
        [copy(el) for _ in 1:N],
        zeros(Int64, N)
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


# TODO typing?
"""
    append!(LogBinner, values)

Adds an array of values to the binner by `push!`ing each element.
"""
function Base.append!(B::LogBinner, values::AbstractArray)
    for value in values
        push!(B, value)
    end
    nothing
end


"""
    push!(LogBinner, value)

Pushes a new value into the Binning Analysis.
"""
function Base.push!(B::LogBinner{N, T}, value::S) where {N, T, S}
    ndims(T) == ndims(S) || throw(DimensionMismatch("Expected $(ndims(T)) dimensions but got $(ndims(S))."))

    _push!(B, 1, value)
end


_square(x) = x^2
_square(x::Complex) = Complex(real(x)^2, imag(x)^2)
_square(x::AbstractArray) = _square.(x)

# recursion, back-end function
function _push!(B::LogBinner{N, T}, lvl::Int64, value::S) where {N, T <: Number, S}
    C = B.compressors[lvl]

    # any value propagating through this function is new to lvl. Therefore we
    # add it to the sums. Note that values pushed to the output arrays are not
    # added here until the array drops to the next level. (New compressors are
    # added)
    B.x_sum[lvl] += value
    B.x2_sum[lvl] += _square(value)
    B.count[lvl] += 1

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
        B::LogBinner{N, T},
        lvl::Int64,
        value::S
    ) where {N, T <: AbstractArray, S}

    C = B.compressors[lvl]
    B.x_sum[lvl] .+= value
    B.x2_sum[lvl] .+= _square(value)
    B.count[lvl] += 1

    if !C.switch
        C.value = value
        C.switch = true
        return nothing
    else
        if lvl == N
            throw(OverflowError("The Binning Analysis has exceeded its maximum capacity."))
        else
            C.switch = false
            _push!(B, lvl+1, 0.5 * (C.value .+ value))
            return nothing
        end
    end
    return nothing
end
