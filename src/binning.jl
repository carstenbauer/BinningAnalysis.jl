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

"""
    LogBinner([T = Float64, N = 32])

Creates a Binning Analysis which can take 2^N-1 value sof type T.
"""
LogBinner(T::Type = Float64, N::Int64 = 32) = LogBinner(zero(T), N)

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
    LogBinner([zero = 0.0, N = 32])

Creates a new Binning Analysis which can take 2^N-1 values of type T. The type
is inherited by the given zero. Returns a Binning Analysis object.

Values can be added using `push!(LogBinner, value)`.
"""
function LogBinner(_zero::T = zero(Float64), N::Int64 = 32) where {T}
    LogBinner{N, T}(
        tuple([Compressor{T}(copy(_zero), false) for i in 1:N]...),
        [copy(_zero) for _ in 1:N],
        [copy(_zero) for _ in 1:N],
        zeros(Int64, N)
    )
end


# TODO typing?
"""
    append!(LogBinner, values)

Adds an array of values to the Binning Analysis by applying push! to each
element.
"""
function append!(B::LogBinner, values::AbstractArray)
    for value in values
        push!(B, value)
    end
    nothing
end


"""
    push!(LogBinner, value)

Pushes a new value into the Binning Analysis.
"""
function push!(B::LogBinner{N, T}, value::T) where {N, T}
    push!(B, 1, value)
end


_square(x) = x^2
_square(x::Complex) = Complex(real(x)^2, imag(x)^2)
_square(x::AbstractArray) = _square.(x)

# recursion, back-end function
function push!(B::LogBinner{N, T}, lvl::Int64, value::T) where {N, T <: Number}
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
            throw(OverflowError("The Binning Analysis ha exceeddd its maximum capacity."))
        else
            # propagate to next lvl
            C.switch = false
            push!(B, lvl+1, 0.5 * (C.value + value))
            return nothing
        end
    end
    return nothing
end

function push!(
        B::LogBinner{N, T},
        lvl::Int64,
        value::T
    ) where {N, T <: AbstractArray}

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
            throw(OverflowError("The Binning Analysis ha exceeddd its maximum capacity."))
        else
            C.switch = false
            push!(B, lvl+1, 0.5 * (C.value .+ value))
            return nothing
        end
    end
    return nothing
end