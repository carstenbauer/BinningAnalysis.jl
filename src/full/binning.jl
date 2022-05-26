# Thin wrapper type, mainly for dispatch
struct FullBinner{T, A <: AbstractVector{T}} <: AbstractBinner{T}
    x::A
end

FullBinner(::Type{T} = Float64) where T = FullBinner(Vector{T}(undef, 0))


@forward FullBinner.x (Base.length, Base.size, Base.lastindex, Base.ndims, Base.iterate,
                        Base.getindex, Base.setindex!, Base.view, Base.axes,
                        Base.resize!, Base.sizehint!, Base.empty!, Base.isempty)

Base.:(!=)(a::FullBinner, b::FullBinner) = a.x != b.x
Base.:(==)(a::FullBinner, b::FullBinner) = a.x == b.x
function Base.isapprox(a::FullBinner, b::FullBinner; kwargs...)
    (length(a) == length(b)) && (isempty(a) || isapprox(a.x, b.x; kwargs...))
end


# Cosmetics

function _print_header(io::IO, B::FullBinner{T,A}) where {T, A}
    print(io, "FullBinner{$(T),$(A)}")
    nothing
end

function _println_body(io::IO, B::FullBinner{T,A}) where {T, A}
    n = length(B)
    println(io)
    print(io, "| Count: ", n)
    if n > 0 && T <: Number
        print(io, "\n| Mean: ", round.(mean(B), digits=5))
        # print(io, "\n| StdError: ", round.(std_error(B), digits=5))
    end
    nothing
end

# short version (shows up in arrays etc.)
Base.show(io::IO, B::FullBinner{T,A}) where {T, A} = print(io, "FullBinner{$(T),$(A)}()")
# verbose version (shows up in the REPL)
Base.show(io::IO, m::MIME"text/plain", B::FullBinner) = (_print_header(io, B); _println_body(io, B))

# We explicitly copy here to avoid falsifying data when a pushed array is 
# overwritten. (For example one might have a temporary array which accumulates
# data before pushing it to a FullBinner. If this array is reused for something
# else and we do not copy here, the pushed element of the FullBinner changes.
# This is not desirable.)
Base.push!(B::FullBinner, x) = push!(B.x, deepcopy(x))
Base.append!(B::FullBinner{<: Number}, x::AbstractArray{<: Number}) = append!(B.x, x)