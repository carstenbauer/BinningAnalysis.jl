using BinningAnalysis
using Test


@testset "All Tests" begin
    @testset "Checking converging data (Real)" begin
        BA = LogBinner()

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
        means = BinningAnalysis.all_means(BA)
        for x in means
            @test x ≈ means[1]
        end
    end

    @testset "Check variance for Complex values" begin
        # NOTE
        # Due to the different (mathematically equivalent) versions of the variance
        # calculated here, the values are onyl approximately the same. (Float error)
        xs = rand(ComplexF64, 1_000_000)
        BA = LogBinner(ComplexF64)

        # Test small set (off by one errors are large here)
        for x in xs[1:10]; push!(BA, x) end
        @test var(BA, 1) ≈ var(xs[1:10])
        @test varN(BA, 1) ≈ var(xs[1:10])/10

        # Test full set
        for x in xs[11:end]; push!(BA, x) end
        @test var(BA, 1) ≈ var(xs)
        @test varN(BA, 1) ≈ var(xs)/1_000_000
    end



    @testset "Checking converging data (Complex)" begin
        BA = LogBinner(ComplexF64)

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
        means = BinningAnalysis.all_means(BA)
        for x in means
            @test x ≈ means[1]
        end
    end



    @testset "Checking converging data (Vector)" begin
        BA = LogBinner(zeros(3))

        # block of maximally correlated values:
        N_corr = 16

        # number of blocks
        N_blocks = 131_072 # 2^17

        for _ in 1:N_blocks
            x = rand(Float64, 3)
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
        means = BinningAnalysis.all_means(BA)
        for i in eachindex(means)
            @test means[i] ≈ means[1]
        end
    end



    @testset "Type promotion" begin
        Bf = LogBinner(zero(1.)) # Float64 LogBinner
        Bc = LogBinner(zero(im)) # Float64 LogBinner

        # Check that this doesn't throw (TODO: is there a better way?)
        @test (append!(Bf, rand(1:10, 10000)); true)
        @test (append!(Bc, rand(10000)); true)
    end


    @testset "Sum-type heuristic" begin
        # numbers
        @test typeof(LogBinner(zero(Int64))) == LogBinner{32,Float64}
        @test typeof(LogBinner(zero(ComplexF16))) == LogBinner{32,ComplexF64}

        # arrays
        @test typeof(LogBinner(zeros(Int64, 2,2))) == LogBinner{32,Matrix{Float64}}
        @test typeof(LogBinner(zeros(ComplexF16, 2,2))) == LogBinner{32,Matrix{ComplexF64}}
    end


    @testset "Basic Base functions" begin
        # numbers
        for T in (Float64, ComplexF64)
            B = LogBinner(T)

            @test length(B) == 0
            @test ndims(B) == 0
            @test isempty(B)
            @test eltype(B) == T

            append!(B, rand(1000))
            @test length(B) == 1000
            @test !isempty(B)

            empty!(B)
            @test length(B) == 0
            @test isempty(B)
        end

        # arrays
        for T in (Float64, ComplexF64)
            B = LogBinner(zeros(T, 2, 3))

            @test length(B) == 0
            @test ndims(B) == 2
            @test isempty(B)
            @test eltype(B) == Array{T, 2}

            append!(B, [rand(T, 2,3) for _ in 1:1000])
            @test length(B) == 1000
            @test !isempty(B)
            empty!(B)
            @test length(B) == 0
            @test isempty(B)
        end
    end


    @testset "Indexing Bounds" begin
        BA = LogBinner(zero(Float64), 1)
        for func in [:var, :varN, :mean, :tau]
            # check if func(BA, 0) throws BoundsError
            # It should as level 1 is now the initial level
            @test_throws BoundsError @eval $func($BA, 0)
            # Check that level 1 exists
            @test (@eval $func($BA, 1); true)
            # Check that level 2 throws BoundsError
            @test_throws BoundsError @eval $func($BA, 2)
        end
    end


    @testset "_select_lvl_for_std_error" begin
        BA = LogBinner()
        # Empty Binner
        @test BinningAnalysis._select_lvl_for_std_error(BA) == 1
        @test isnan(std_error(BA, BinningAnalysis._select_lvl_for_std_error(BA)))

        # One Element should still return NaN (due to 1/(n-1))
        push!(BA, rand())
        @test BinningAnalysis._select_lvl_for_std_error(BA) == 1
        @test isnan(std_error(BA, BinningAnalysis._select_lvl_for_std_error(BA)))

        # Two elements should return some value
        push!(BA, rand())
        @test BinningAnalysis._select_lvl_for_std_error(BA) == 1
        @test !isnan(std_error(BA, BinningAnalysis._select_lvl_for_std_error(BA)))

        # same behavior up to (including) 63 values (31 binned in first binned lvl)
        append!(BA, rand(61))
        @test BinningAnalysis._select_lvl_for_std_error(BA) == 1
        @test !isnan(std_error(BA, BinningAnalysis._select_lvl_for_std_error(BA)))

        # at 64 or more values, the lvl should be increasing
        push!(BA, rand())
        @test BinningAnalysis._select_lvl_for_std_error(BA) == 2
        @test !isnan(std_error(BA, BinningAnalysis._select_lvl_for_std_error(BA)))
    end
end
