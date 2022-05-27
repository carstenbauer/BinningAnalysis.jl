# This has the same job as the LogBinner Compressor
mutable struct EPCompressor{T}
    # One value per input argument
    values::Vector{T}
    switch::Bool
end

function Base.:(==)(a::EPCompressor, b::EPCompressor; kwargs...)
    # overwrite mode or same value
    (a.switch == b.switch) && ((a.switch == false) || (a.values == b.values))
end
function Base.isapprox(a::EPCompressor, b::EPCompressor; kwargs...)
    (a.switch == b.switch) && (a.switch || isapprox(a.values, b.values; kwargs...))
end

struct ErrorPropagator{T, N} <: AbstractBinner{T}
    compressors::NTuple{N, EPCompressor{T}}
    # ∑x
    sums1D::NTuple{N, Vector{T}}
    # ∑xy
    sums2D::NTuple{N, Matrix{T}}

    count::Vector{Int64}
end


# Overload some basic Base functions
Base.length(ep::ErrorPropagator) = ep.count[1]
Base.ndims(ep::ErrorPropagator{T,N}) where {T,N} = ndims(eltype(ep))
Base.isempty(ep::ErrorPropagator) = length(ep) == 0

Base.:(!=)(a::ErrorPropagator, b::ErrorPropagator) = !(a == b)
function Base.:(==)(a::ErrorPropagator, b::ErrorPropagator)
    length(a.count) > length(b.count) && return b == a
    
    for i in eachindex(a.count)
        a.compressors[i] == b.compressors[i] || return false
        a.sums1D[i] == b.sums1D[i] || return false
        a.sums2D[i] == b.sums2D[i] || return false
        a.count[i] == b.count[i] || return false
    end

    return true
end

function Base.isapprox(a::ErrorPropagator, b::ErrorPropagator; kwargs...)
    length(a.count) > length(b.count) && return b == a
    
    for i in eachindex(a.count)
        isapprox(a.compressors[i], b.compressors[i]; kwargs...) || return false
        isapprox(a.sums1D[i], b.sums1D[i]; kwargs...) || return false
        isapprox(a.sums2D[i], b.sums2D[i]; kwargs...) || return false
        isapprox(a.count[i], b.count[i]; kwargs...) || return false
    end

    return true
end




function _print_header(io::IO, ::ErrorPropagator{T,N}) where {T,N}
    print(io, "ErrorPropagator{$(T),$(N)}")
    nothing
end

function _println_body(io::IO, ep::ErrorPropagator{T,N}) where {T,N}
    n = length(ep)
    println(io)
    print(io, "| Count: ", n)
    if n > 0 && ndims(ep) == 0
        print(io, "\n| Means: ", round.(means(ep), digits=5))
        print(io, "\n| StdErrors: ", round.(std_errors(ep), digits=5))
    end
    nothing
end

# short version (shows up in arrays etc.)
function Base.show(io::IO, ep::ErrorPropagator{T,N}) where {T,N}
    print(io, "ErrorPropagator{$(T),$(N)}()")
end
# verbose version (shows up in the REPL)
function Base.show(io::IO, m::MIME"text/plain", ep::ErrorPropagator)
    (_print_header(io, ep); _println_body(io, ep))
end




"""
    empty!(ep::ErrorPropagator)

Clear the error propagator, i.e. reset it to its inital state.
"""
function Base.empty!(ep::ErrorPropagator)
    !isempty(ep) || return

    # get zero
    z = zero(eltype(eltype(ep)))

    # reset sums to all zeros
    @inbounds for i in eachindex(ep.sums1D)
        if ndims(ep) == 0 # compile time
            # numbers
            ep.sums1D[i] .= z
            ep.sums2D[i] .= z
        else
            # arrays
            for j in eachindex(ep.sums1D[i])
                ep.sums1D[i][j] .= z
            end
            for j in eachindex(ep.sums2D[i])
                ep.sums2D[i][j] .= z
            end
        end
    end


    # reset counts
    fill!(ep.count, 0)

    # reset compressors
    for c in ep.compressors
        if ndims(ep) == 0 # compile time
            c.values .= z
        else
            for i in eachindex(c.values)
                c.values[i] .= z
            end
        end
        c.switch = false
    end

    nothing
end


"""
    capacity(ep)

Capacity of the error propagator, i.e. how many values can be handled before
overflowing.
"""
capacity(ep::ErrorPropagator{T,N}) where {T,N} = _nlvls2capacity(N)
nlevels(ep::ErrorPropagator{T,N}) where {T,N} = N



"""
    ErrorPropagator([::Type{T}; N_args, capacity::Int])

Creates an `ErrorPropagator` which can handle (at least) `capacity` many values
for each of `N_args` inputs of type `T`.

The default is `T = Float64`, `N_args = 2` and `capacity = 2^32-1 ≈ 4e9`.
"""
function ErrorPropagator(::Type{T} = Float64; N_args::Int64 = 2, kw...) where T
    ErrorPropagator([zero(T) for _ in 1:N_args]...; kw...)
end


"""
    ErrorPropagator(zero_element::T...[; capacity::Int])

Creates a new `ErrorPropagator` which can take (at least) `capacity` many
values for each input of type `T`. The type and the size are inherited from the
given `zero_element`s, which must exclusively contain zeros.

Values can be added using `push!` and `append!`.

---

    ErrorPropagator(timeseries::AbstractVector{T}...)

Creates a new `ErrorPropagator` and adds all elements from each given timeseries.
"""
function ErrorPropagator(
            xs::T...;
            capacity::Int64 = _nlvls2capacity(32)
        ) where {T <: Union{Number, AbstractArray}}

    # check keyword args
    capacity <= 0 && throw(ArgumentError("`capacity` must be finite and positive."))

    # got_timeseries = didn't receive a zero && is a vector
    # got_timeseries = count(!iszero, x[1]) > 0 && ndims(T) == 1
    got_timeseries = any(count(!iszero, x) > 0 && ndims(T) == 1 for x in xs)

    if got_timeseries
        # xs[i] = time series
        N = _capacity2nlvls(length(xs[1]))
        S = _sum_type_heuristic(eltype(T), ndims(xs[1][1]))
        el = zero(xs[1][1])
    else
        # xs[i] = zero_element
        N = _capacity2nlvls(capacity)
        S = _sum_type_heuristic(T, ndims(xs[1]))
        el = xs[1]
    end

    ep = ErrorPropagator{S, N}(
        tuple([
            EPCompressor{S}([copy(el) for _ in 1:length(xs)], false)
            for __ in 1:N
        ]...),
        tuple([
            [copy(el) for _ in 1:length(xs)]
            for __ in 1:N
        ]...),
        tuple([
            [copy(el) for _ in 1:length(xs), __ in 1:length(xs)]
            for __ in 1:N
        ]...),
        zeros(Int64, N)
    )

    got_timeseries && append!(ep, xs...)

    return ep
end


# TODO typing?
"""
    append!(ErrorPropagator, values...)

Adds multiple arrays of values to the Error Propagator by `push!`ing each element.
"""
function Base.append!(ep::ErrorPropagator, values::AbstractArray...)
    # TODO does this perform well?
    @inbounds for args in zip(values...)
        _push!(ep, 1, args)
    end
    nothing
end


"""
    push!(ep::ErrorPropagator, args...)

Pushes a set of argumentes `args` into a given error propagator. Note that the
number of arguments must match the initially defined `N_arguments` of the
error propagator.
"""
@inline function Base.push!(ep::ErrorPropagator{T, N}, args::T...) where {T, N}
    # @boundscheck allows you to define a boundscheck that can be omitted with
    # @inbounds
    # but that requires an @inline
    @boundscheck begin
        (length(args) == length(ep.sums1D[1])) || throw(DimensionMismatch(
            "Number of arguments $(length(args)) does not match the number " *
            "of arguments accepted by the error propagator (N_arguments = " *
            "$(length(ep.sums1D[1])))"
        ))
    end
    @inbounds _push!(ep, 1, args)
end


@inline _mult(x, y) = x * y
@inline _mult(x::Complex, y::Complex) = x * conj(y)
@inline _mult(x::AbstractArray, y::AbstractArray) = _mult.(x, y)

# recursion, back-end function
@inline function _push!(ep::ErrorPropagator{T, N}, lvl::Int64, args) where {T, N}
    @boundscheck begin
        (length(args) == length(ep.sums1D[1])) || throw(DimensionMismatch(
            "Number of arguments $(length(args)) does not match the number " *
            "of arguments accepted by the error propagator (N_arguments = " *
            "$(length(ep.sums1D[1])))"
        ))
    end
    C = ep.compressors[lvl]

    # any value propagating through this function is new to lvl. Therefore we
    # add it to the sums. Note that values pushed to the output arrays are not
    # added here until the array drops to the next level. (New compressors are
    # added)
    @simd for i in eachindex(ep.sums1D[lvl])
        ep.sums1D[lvl][i] += args[i]
        for j in eachindex(ep.sums1D[lvl])
            ep.sums2D[lvl][i, j] += _mult(args[i], args[j])
        end
    end
    ep.count[lvl] += 1

    if !C.switch
        # Compressor has space -> save value
        C.values .= args
        C.switch = true
        return nothing
    else
        # Do averaging
        if lvl == N
            # No more propagation possible -> throw error
            throw(OverflowError("The Error Propagator has exceeded its maximum capacity."))
        else
            # propagate to next lvl
            C.switch = false
            @inbounds _push!(ep, lvl+1, 0.5 * (C.values .+ args))
            return nothing
        end
    end

    return nothing
end
