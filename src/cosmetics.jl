function _println_header(io::IO, B::LogBinner{N,T}) where {N, T}
    println("LogBinner{$(N),$(T)}")
end

function _println_body(io::IO, B::LogBinner{N,T}) where {N, T}
    n = B.count[1]
    print("| Count: ", n)
    if n > 0 && ndims(B) == 0
        print("\n| Mean: ", round.(mean(B), digits=5))
        print("\n| StdError: ", round.(std_error(B), digits=5))
    end
end

Base.show(io::IO, B::LogBinner{N,T}) where {N, T} = begin
    _println_header(io, B)
    _println_body(io, B)
    nothing
end
Base.show(io::IO, m::MIME"text/plain", B::LogBinner{N,T}) where {N, T} = print(io, B)

Base.summary(io::IO, B::LogBinner{N,T}) where {N, T} = _println_header(io, B)
Base.summary(B::LogBinner{N,T}) where {N, T} = summary(stdout, B)