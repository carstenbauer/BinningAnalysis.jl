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

Base.summary(io::IO, F::FullBinner) = nothing#_print_header(io, F)
Base.summary(F::FullBinner) = nothing#summary(stdout, F)