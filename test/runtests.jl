using BinningAnalysis
@static if VERSION < v"0.7"
    using Base.Test
else
    using Test
end

# write your own tests here
@testset "Checking converging data (Real)" begin
    BA = BinnerA()

    # block of maximally correlated values:
    N_corr = 16

    # number of blocks
    N_blocks = 131_072 # 2^17

    for _ in 1:N_blocks
        x = rand()
        for __ in 1:N_corr
            push!(BA, x)
        end
    end

    # BA must diverge until 16 values are binned
    for lvl in 1:4
        @test !has_converged(BA, lvl)
    end

    # Afterwards it must converge (for a while)
    for lvl in 5:8
        @test has_converged(BA, lvl)
    end
    # Later values may fluctuate due to small samples / small threshold in
    # has_converged

    # means should be consistent
    means = all_means(BA)
    for x in means
        @test x ≈ means[1]
    end
end

@testset "Check variance for Complex values" begin
    # NOTE
    # Due to the different (mathematically equivalent) versions of the variance
    # calculated here, the values are onyl approximately the same. (Float error)
    xs = rand(ComplexF64, 1_000_000)
    BA = BinnerA(ComplexF64)

    # Test small set (off by one errors are large here)
    for x in xs[1:10]; push!(BA, x) end
    @test var(BA, 0) ≈ var(xs[1:10])
    @test varN(BA, 0) ≈ var(xs[1:10])/10

    # Test full set
    for x in xs[11:end]; push!(BA, x) end
    @test var(BA, 0) ≈ var(xs)
    @test varN(BA, 0) ≈ var(xs)/1_000_000
end


# write your own tests here
@testset "Checking converging data (Complex)" begin
    BA = BinnerA(ComplexF64)

    # block of maximally correlated values:
    N_corr = 16

    # number of blocks
    N_blocks = 131_072 # 2^17

    for _ in 1:N_blocks
        x = rand(ComplexF64)
        for __ in 1:N_corr
            push!(BA, x)
        end
    end

    # BA must diverge until 16 values are binned
    for lvl in 1:4
        @test !has_converged(BA, lvl)
    end

    # Afterwards it must converge (for a while)
    for lvl in 5:8
        @test has_converged(BA, lvl)
    end
    # Later values may fluctuate due to small samples / small threshold in
    # has_converged

    # means should be consistent
    means = all_means(BA)
    for x in means
        @test x ≈ means[1]
    end
end
