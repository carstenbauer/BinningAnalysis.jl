@testset "Cosmetics (show, print, etc.)" begin
    B = PreBinner(LogBinner(), 10)

    # empty binner
    io = IOBuffer()
    println(io, B) # compact
    show(io, MIME"text/plain"(), B) # full

    l = String(take!(io))
    @test l == "PreBinner{Float64, LogBinner}()\nPreBinned (10) LogBinner{Float64,32}\n| Count: 10 × 0 + 0 (0)"
    @test length(readlines(io)) == 0


    # filled binner
    StableRNGs.seed!(rng, 123)
    append!(B, rand(rng, 1000))
    show(io, MIME"text/plain"(), B)

    l = String(take!(io))
    @test l == "PreBinned (10) LogBinner{Float64,32}\n| Count: 10 × 100 + 10 (1010)\n| Mean: 0.49387\n| StdError: 0.00946"
    @test length(readlines(io)) == 0
    close(io);
end

for wrapped in (LogBinner, FullBinner)
    @testset "Wrapped $wrapped" begin
        B = PreBinner(wrapped(), 10)
        B2 = PreBinner(wrapped(), 10)
        
        @test B.binner isa wrapped
        @test length(B) == 0
        wrapped == LogBinner && (@test count(B) == 0)
        @test ndims(B) == 0
        @test isempty(B)
        @test eltype(B) == Float64
        @test B == B2
        @test B ≈ B2
        @test !(B != B2)
        
        @test B.N == 10
        @test B.n == 0

        
        StableRNGs.seed!(rng, 123)
        xs = rand(rng, 789)
        append!(B, xs)

        @test !(B == B2)
        @test !(B ≈ B2)
        @test B != B2
        @test B.n == 9
        @test length(B) == 789
        @test length(B.binner) == 78

        @test mean(B) ≈ mean(xs) # Float errors
        @test std_error(B) == std_error(B.binner)
        @test BinningAnalysis._reliable_level(B) == 1 + BinningAnalysis._reliable_level(B.binner)
    end
end