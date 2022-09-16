mutable struct PreBinner{T, BT, AT} <: AbstractBinner{T}
    binner::BT
    buffer::T
    accumulator::AT
    n::Int # 0..N-1, push and reset on N
    N::Int
end

function PreBinner(B::AbstractBinner{T}, N::Int) where {T}
    N > 0 || error("Must average at least 1 value.")

    if T <: Number
        buffer = zero(T)
    else
        dummy = mean(B)
        buffer = zeros(eltype(dummy), size(dummy))
    end

    return PreBinner(B, buffer, Variance{T}(deepcopy(buffer)), 0, N)
end

################################################################################
### Util
################################################################################

Base.count(B::PreBinner, lvl::Int = 1) = B.N * count(B.binner, max(1, lvl-1)) + (lvl == 1) * B.n
Base.length(B::PreBinner) = count(B, 1)
Base.ndims(B::PreBinner) = ndims(B.binner)
Base.isempty(B::PreBinner) = length(B) == 0

function Base.isapprox(a::PreBinner{T, BT}, b::PreBinner{T, BT}; kwargs...) where {T, BT}
    ((a.n == b.n) && (a.N == b.N)) || return false
    isapprox(a.accumulator, b.accumulator; kwargs...) || return false
    return isapprox(a.binner, b.binner; kwargs...)
end

function Base.:(==)(a::PreBinner{T, BT}, b::PreBinner{T, BT}) where {T, BT}
    ((a.n == b.n) && (a.N == b.N)) || return false
    a.accumulator == b.accumulator || return false
    return a.binner == b.binner
end

Base.:(!=)(a::PreBinner, b::PreBinner) = !(a == b)


# short version (shows up in arrays etc.)
function Base.show(io::IO, B::PreBinner{T, BT}) where {T, BT}
    print(io, "PreBinner{$(T), $(BT.name.name)}()")
end
# verbose version (shows up in the REPL)
function Base.show(io::IO, m::MIME"text/plain", B::PreBinner)
    print(io, "PreBinned ($(B.N)) ")
    _print_header(io, B.binner)
    
    n = length(B.binner)
    println(io)
    print(io, "| Count: ", B.N, " × ", n, " + ", B.n, " (", length(B), ")")
    if n > 0 && ndims(B) == 0
        print(io, "\n| Mean: ", round.(mean(B), digits=5))
        print(io, "\n| StdError: ", round.(std_error(B), digits=5))
    end
    nothing
end


function Base.empty!(B::PreBinner)
    empty!(B.binner)
    B.n = 0
    return B
end

nlevels(B::PreBinner) = 1 + nlevels(B.binner)
capacity(B::PreBinner) = B.N * capacity(B.binner) + B.n

################################################################################
### Pushing
################################################################################

function Base.push!(B::PreBinner{T,N}, value::S) where {N, T, S}
    ndims(T) == ndims(S) || throw(DimensionMismatch("Expected $(ndims(T)) dimensions but got $(ndims(S))."))

    # 0/N -> 1 write
    # N-1 -> N add, div, push
    # else add
    n = B.n % B.N
    _push!(B.accumulator, value)
    if n == 0
        _set!(B, value)
        B.n = 0
    elseif n == B.N - 1
        _push!(B, value)
    else
        _add!(B, value)
    end
    B.n += 1
    return nothing
end

_set!(B::PreBinner, value::Number) = B.buffer = value
_set!(B::PreBinner, value::AbstractArray) = B.buffer .= value
_add!(B::PreBinner, value::Number) = B.buffer += value
_add!(B::PreBinner, value::AbstractArray) = B.buffer .+= value
function _push!(B::PreBinner, value::Number)
    B.buffer += value
    push!(B.binner, B.buffer / B.N)
end
function _push!(B::PreBinner, value::AbstractArray)
    B.buffer .+= value
    B.buffer ./= B.N
    push!(B.binner, B.buffer)
end

################################################################################
### Statistics
################################################################################

function mean(B::PreBinner, lvl = 1)
    lvl == 1 ? mean(B.accumulator) : mean(B.binner, max(1, lvl-1))
end

_reliable_level(B::PreBinner) = 1 + _reliable_level(B.binner)

function var(B::PreBinner, lvl =  _reliable_level(B))
    lvl == 1 ? var(B.accumulator) : var(B.binner, max(1, lvl-1))
end

function varN(B::PreBinner, lvl = _reliable_level(B))
    lvl == 1 ? varN(B.accumulator) : varN(B.binner, max(1, lvl-1))
end