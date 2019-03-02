@testset "Constructors and basic properties" begin
    # numbers
    for T in (Float64, ComplexF64)
        B = LogBinner(T)

        @test length(B) == 0
        @test ndims(B) == 0
        @test isempty(B)
        @test eltype(B) == T
        @test capacity(B) == 2^32 - 1
        @test BinningAnalysis.nlevels(B) == 32

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
        @test capacity(B) == 2^32 - 1
        @test BinningAnalysis.nlevels(B) == 32

        append!(B, [rand(T, 2,3) for _ in 1:1000])
        @test length(B) == 1000
        @test !isempty(B)
        empty!(B)
        @test length(B) == 0
        @test isempty(B)
    end

    # Constructor arguments
    B = LogBinner(capacity=12345)
    @test capacity(B) == 16383
    @test_throws ArgumentError LogBinner(capacity=0)
    @test_throws ArgumentError LogBinner(capacity=-1)


    # Test overflow
    B = LogBinner(capacity=1)
    push!(B, 1.0); push!(B, 2.0)
    @test !isempty(B.overflow)

    B = LogBinner(zeros(2,2), capacity=1)
    push!(B, rand(2,2)); push!(B, rand(2,2))
    @test !isempty(B.overflow)

    # time series constructor (#26)
    x = rand(10)
    B = LogBinner(x)
    @test length(B) == 10
    @test mean(B) == mean(x)

    x = [rand(2,3) for _ in 1:5]
    B = LogBinner(x)
    @test length(B) == 5
    @test mean(B) == mean(x)

    # Test LogBinner(::LogBinner)
    for T in [Float64, ComplexF32]
        xs = rand(T, 1000)
        small_Binner = LogBinner(T, capacity = 200)
        large_Binner = LogBinner(T)
        append!(small_Binner, xs)
        append!(large_Binner, xs)
        ext_Binner = LogBinner(small_Binner)

        @test all(ext_Binner.x_sum .== large_Binner.x_sum)
        @test all(ext_Binner.x2_sum .== large_Binner.x2_sum)
        @test all(ext_Binner.count .== large_Binner.count)
        @test all(ext_Binner.overflow .== large_Binner.overflow)
        @test map(ext_Binner.compressors, large_Binner.compressors) do c1, c2
            (c1.value == c2.value) && (c1.switch == c2.switch)
        end |> all
    end

    xs = [rand(2, 2) for _ in 1:1000]
    small_Binner = LogBinner(zeros(2, 2), capacity = 200)
    large_Binner = LogBinner(zeros(2, 2))
    append!(small_Binner, xs)
    append!(large_Binner, xs)
    ext_Binner = LogBinner(small_Binner)
    
    @test all(ext_Binner.x_sum .== large_Binner.x_sum)
    @test all(ext_Binner.x2_sum .== large_Binner.x2_sum)
    @test all(ext_Binner.count .== large_Binner.count)
    @test all(ext_Binner.overflow .== large_Binner.overflow)
    @test map(ext_Binner.compressors, large_Binner.compressors) do c1, c2
        (c1.value == c2.value) && (c1.switch == c2.switch)
    end |> all
end


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

@testset "Check variance for complex values" begin
    # NOTE
    # Due to the different (mathematically equivalent) versions of the variance
    # calculated here, the values are onyl approximately the same. (Float error)
    Random.seed!(1234)
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

    # all_* methods
    @test isapprox(all_vars(BA), [0.16671474067121222, 0.08324845751233179, 0.041527133392489035, 0.020847602123934883, 0.010430741538377142, 0.005111097271805531, 0.0025590988213273214, 0.001283239131297187, 0.0006322480081128456, 0.0003060164750540162, 0.00015750782337442537, 8.006142368921498e-5, 3.810111634139357e-5, 1.80535512880331e-5, 1.0002438211476061e-5, 5.505102193326117e-6, 2.8788397929968568e-6, 1.7242475507384114e-6, 7.900888818745955e-7])
    @test isapprox(all_varNs(BA), zero(all_varNs(BA)), atol=1e-6)
    @test isapprox(all_taus(BA), [0.0, -0.00065328850247931, -0.0018180968845809553, 0.00019817179932868356, 0.0005312186016332987, -0.009476150581268994, -0.008794711536776634, -0.007346737569564443, -0.014542478848703244, -0.030064159934323653, -0.01599670814224563, -0.007961042363178128, -0.03167873601168558, -0.056188229083248886, -0.008218660661725774, 0.05035147373711113, 0.0756019296606737, 0.2387501479629205, 0.289861051172009])
    @test isapprox(all_std_errors(BA), [0.0004083071646092097, 0.0004080403350462593, 0.00040756414657076514, 0.0004083880715587553, 0.0004085240073900606, 0.00040441947616030687, 0.00040470028980091994, 0.0004052963382191276, 0.00040232555162611277, 0.00039584146247874917, 0.00040172249945971054, 0.00040504357104527986, 0.00039516087376314515, 0.0003846815937765092, 0.00040493752222959487, 0.0004283729758565588, 0.00043808977717638777, 0.0004963074437049236, 0.0005131890106236348])
end



@testset "Check variance for complex vectors" begin
    Random.seed!(1234)
    xs = [rand(ComplexF64, 3) for _ in 1:1_000_000]
    BA = LogBinner(zeros(ComplexF64, 3))

    # Test small set (off by one errors are large here)
    for x in xs[1:10]; push!(BA, x) end
    @test var(BA, 1) ≈ var(xs[1:10])
    @test varN(BA, 1) ≈ var(xs[1:10])/10

    # Test full set
    for x in xs[11:end]; push!(BA, x) end
    @test var(BA, 1) ≈ var(xs)
    @test varN(BA, 1) ≈ var(xs)/1_000_000

    # all_std_errors for <:AbstractArray
    @test all(isapprox.(all_std_errors(BA), Ref(zeros(3)), atol=1e-2))
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



@testset "Indexing Bounds" begin
    BA = LogBinner(zero(Float64); capacity=1)
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



@testset "_reliable_level" begin
    BA = LogBinner()
    # Empty Binner
    @test BinningAnalysis._reliable_level(BA) == 1
    @test isnan(std_error(BA, BinningAnalysis._reliable_level(BA)))

    # One Element should still return NaN (due to 1/(n-1))
    push!(BA, rand())
    @test BinningAnalysis._reliable_level(BA) == 1
    @test isnan(std_error(BA, BinningAnalysis._reliable_level(BA)))

    # Two elements should return some value
    push!(BA, rand())
    @test BinningAnalysis._reliable_level(BA) == 1
    @test !isnan(std_error(BA, BinningAnalysis._reliable_level(BA)))

    # same behavior up to (including) 63 values (31 binned in first binned lvl)
    append!(BA, rand(61))
    @test BinningAnalysis._reliable_level(BA) == 1
    @test !isnan(std_error(BA, BinningAnalysis._reliable_level(BA)))

    # at 64 or more values, the lvl should be increasing
    push!(BA, rand())
    @test BinningAnalysis._reliable_level(BA) == 2
    @test !isnan(std_error(BA, BinningAnalysis._reliable_level(BA)))
end




@testset "Cosmetics (show, print, etc.)" begin
    B = LogBinner();
    # empty binner
    oldstdout = stdout
    (read_pipe, write_pipe) = redirect_stdout()
    println(B) # compact
    show(write_pipe, MIME"text/plain"(), B) # full
    redirect_stdout(oldstdout);
    close(write_pipe);

    # compact
    @test readline(read_pipe) == "LogBinner{32,Float64}()"
    # full
    @test readline(read_pipe) == "LogBinner{32,Float64}"
    @test readline(read_pipe) == "| Count: 0"
    @test length(readlines(read_pipe)) == 0
    close(read_pipe);

    # filled binner
    Random.seed!(1234)
    append!(B, rand(1000))
    (read_pipe, write_pipe) = redirect_stdout()
    show(write_pipe, MIME"text/plain"(), B)
    redirect_stdout(oldstdout);
    close(write_pipe);
    @test readline(read_pipe) == "LogBinner{32,Float64}"
    @test readline(read_pipe) == "| Count: 1000"
    @test readline(read_pipe) == "| Mean: 0.49685"
    @test readline(read_pipe) == "| StdError: 0.00733"
    @test length(readlines(read_pipe)) == 0
    close(read_pipe);
end
