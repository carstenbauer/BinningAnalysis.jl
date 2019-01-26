# This is a Node in the Binning Analysis tree that averages two values. There
# is one of these for each binning level. When two values should be
# compressed, this is done immediately, so that only one value needs to be saved.
# switch indicates whether value should be written to or averaging should happen.
mutable struct Compressor{T}
    value::T
    switch::UInt8
end


struct BinnerA{N, T}
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
    BinnerA([T = Float64, N = 32])

Creates a new Binning Analysis which can take 2^N-1 values of type T. Returns a
Binning Analysis object.

Values can be added using `push!(BinnerA, value)`.
"""
function BinnerA(T::Type = Float64, N::Int64 = 32)
    BinnerA{N, T}(
        tuple([Compressor{T}(zero(T), UInt8(0)) for i in 1:N]...),
        zeros(T, N),
        zeros(T, N),
        zeros(Int64, N)
    )
end


# TODO typing?
"""
    append!(BinnerA, values)

Adds an array of values to the Binning Analysis by applying push! to each
element.
"""
function append!(B::BinnerA, values::AbstractArray)
    for value in values
        push!(B, value)
    end
    nothing
end


"""
    push!(BinnerA, value)

Pushes a new value into the Binning Analysis.
"""
function push!(B::BinnerA{N, T}, value::T) where {N, T}
    push!(B, 1, value)
end


_square(x) = x^2
_square(x::Complex) = Complex(real(x)^2, imag(x)^2)
_square(x::AbstractArray) = map(_square, x)

# recursion, back-end function
function push!(B::BinnerA{N, T}, lvl::Int64, value::T) where {N, T}
    C = B.compressors[lvl]

    # any value propagating through this function is new to lvl. Therefore we
    # add it to the sums. Note that values pushed to the output arrays are not
    # added here until the array drops to the next level. (New compressors are
    # added)
    B.x_sum[lvl] += value
    B.x2_sum[lvl] += _square(value)
    B.count[lvl] += 1

    if C.switch == 0
        # Compressor has space -> save value
        C.value = value
        C.switch = 1
        return nothing
    else
        # Do averaging
        if lvl == N
            # No more propagation possible -> throw error
            error("The Binning Analysis ha exceeddd its maximum capacity.")
        else
            # propagate to next lvl
            C.switch = 0
            push!(B, lvl+1, 0.5 * (C.value + value))
            return nothing
        end
    end
    return nothing
end
