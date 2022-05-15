@testset "Constructors and basic properties" begin
    # numbers
    for T in (Float64, ComplexF64), V in (BinningAnalysis.Variance, BinningAnalysis.FastVariance)
        B = LogBinner(T; accumulator=V)
        B2 = LogBinner(T; accumulator=V)

        @test length(B) == 0
        @test ndims(B) == 0
        @test isempty(B)
        @test eltype(B) == T
        @test capacity(B) == 2^32 - 1
        @test BinningAnalysis.nlevels(B) == 32
        @test B == B2
        @test B ≈ B2
        @test !(B != B2)

        B3 = LogBinner(B, capacity=10_000)
        @test B3 == B
        @test B3 ≈ B
        @test capacity(B3) == 16383

        buffer = rand(rng, T, 1000)
        append!(B, buffer)
        @test length(B) == 1000
        @test !isempty(B)
        @test !(B == B2)
        @test !(B ≈ B2)
        @test B != B2

        @test B3 != B
        @test isempty(B3)
        B3 = LogBinner(B, capacity=1000)
        @test B3 == B
        @test B3 ≈ B
        @test !isempty(B3)

        append!(B2, buffer)
        @test B == B2
        @test B ≈ B2
        @test !(B != B2)

        append!(B, rand(rng, T, 24))
        @test_throws OverflowError LogBinner(B, capacity=1000)

        empty!(B)
        @test length(B) == 0
        @test isempty(B)
    end

    # arrays
    for T in (Float64, ComplexF64), V in (BinningAnalysis.Variance, BinningAnalysis.FastVariance)
        B = LogBinner(zeros(T, 2, 3), accumulator=V)
        B2 = LogBinner(zeros(T, 2, 3), accumulator=V)

        @test length(B) == 0
        @test ndims(B) == 2
        @test isempty(B)
        @test eltype(B) == Array{T, 2}
        @test capacity(B) == 2^32 - 1
        @test BinningAnalysis.nlevels(B) == 32
        @test B == B2
        @test B ≈ B2
        @test !(B != B2)

        B3 = LogBinner(B, capacity=10_000)
        @test B3 == B
        @test B3 ≈ B
        @test capacity(B3) == 16383

        buffer = [rand(rng, T, 2, 3) for _ in 1:1000]
        append!(B, buffer)
        @test length(B) == 1000
        @test !isempty(B)
        @test !(B == B2)
        @test !(B ≈ B2)
        @test B != B2

        @test B3 != B
        @test isempty(B3)
        B3 = LogBinner(B, capacity=1000)
        @test B3 == B
        @test B3 ≈ B
        @test !isempty(B3)

        append!(B2, buffer)
        @test B == B2
        @test B ≈ B2
        @test !(B != B2)

        append!(B, [rand(rng, T, 2, 3) for _ in 1:24])
        @test_throws OverflowError LogBinner(B, capacity=1000)

        empty!(B)
        @test length(B) == 0
        @test isempty(B)
    end

    # Constructor arguments
    B = LogBinner(capacity=12345)
    @test capacity(B) == 16383
    @test_throws ArgumentError LogBinner(capacity=0)
    @test_throws ArgumentError LogBinner(capacity=-1)


    # Test error on overflow
    B = LogBinner(capacity=1)
    push!(B, 1.0)
    @test_throws OverflowError push!(B, 2.0)

    B = LogBinner(zeros(2,2), capacity=1)
    push!(B, rand(rng, 2,2))
    @test_throws OverflowError push!(B, rand(rng, 2,2))

    # time series constructor (#26)
    x = rand(rng, 10)
    B = LogBinner(x)
    @test length(B) == 10
    @test mean(B) ≈ mean(x)

    # Test equality of different sizes
    B2 = LogBinner()
    @test !(B == B2)
    @test !(B ≈ B2)
    @test !(B2 == B)
    @test !(B2 ≈ B)
    @test B != B2
    @test B2 != B
    append!(B2, x)
    @test B == B2
    @test B ≈ B2
    @test B2 == B
    @test B2 ≈ B
    @test !(B != B2)
    @test !(B2 != B)

    # Again for Array inputs
    x = [rand(rng, 2,3) for _ in 1:5]
    B = LogBinner(x)
    @test length(B) == 5
    @test mean(B) ≈ mean(x)

    B2 = LogBinner(zeros(2,3))
    @test !(B == B2)
    @test !(B ≈ B2)
    @test !(B2 == B)
    @test !(B2 ≈ B)
    @test B != B2
    @test B2 != B
    append!(B2, x)
    @test B == B2
    @test B ≈ B2
    @test B2 == B
    @test B2 ≈ B
    @test !(B != B2)
    @test !(B2 != B)
end



@testset "Checking converging data (Real)" begin
    BA = LogBinner()

    # block of maximally correlated values:
    N_corr = 16

    # number of blocks
    N_blocks = 131_072 # 2^17

    for _ in 1:N_blocks
        x = rand(rng, )
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
    StableRNGs.seed!(rng, 123)
    xs = rand(rng, ComplexF64, 1_000_000)
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
    @test isapprox(all_vars(BA), [0.1665847540639013, 0.08321852969184897, 0.0416386170603128, 0.02080121032765401, 0.010404232627786522, 0.005208087488833146, 0.0025893712817839105, 0.0012979310682925164, 0.0006498645790020864, 0.0003304988207069092, 0.00016135937775941, 8.244734583734034e-5, 4.1904380844737295e-5, 2.1161425558227823e-5, 1.0185807653474429e-5, 5.794287254586299e-6, 2.4120659277772666e-6, 2.132350340200331e-6, 2.6572093711370505e-7])
    @test isapprox(all_varNs(BA), zero(all_varNs(BA)), atol=1e-6)
    @test isapprox(all_taus(BA), [0.0, -0.0004433019126909299, -9.090214413765008e-5, -0.0005254725849703767, -0.0003512686979507129, 0.00022224596475073355, -0.0025962520837867764, -0.001317479480883732, -0.0006266293029093539, 0.007928314422850802, -0.0037744232961079427, 0.0070976636615537325, 0.01547113876712125, 0.020618795006264512, 0.001187681531748308, 0.07971364057757968, -0.01734960354334969, 0.4143137944503138, -0.23414840330888914])
    @test isapprox(all_std_errors(BA), [0.00040814795609423463, 0.00040796698320292777, 0.0004081108528834429, 0.00040793342915386584, 0.00040800456130365054, 0.0004082386552528566, 0.00040708692196405705, 0.00040760987484761305, 0.00040789211844220833, 0.00041137115440979214, 0.0004066045146218508, 0.00041103464473798035, 0.00041441436480546865, 0.0004164784602525399, 0.0004086324183611065, 0.00043948027087255004, 0.00040100423337559803, 0.0005519252459993797, 0.0002976132485367013])

    @test isapprox(tau(BA), 0.001187681531748308)
    @test isapprox(std_error(BA), 0.0004086324183611065)
end



@testset "Check variance for complex vectors" begin
    StableRNGs.seed!(rng, 123)
    xs = [rand(rng, ComplexF64, 3) for _ in 1:1_000_000]
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

    @test all(isapprox.(tau(BA), [0.025099907279786615, 0.07567404578645565, 0.021586917849341525], atol=1e-6))
    @test all(isapprox.(std_error(BA), [0.000418454645620472, 0.0004382193124499972, 0.00041696433722483093], atol=1e-6))
end



@testset "Checking converging data (Complex)" begin
    BA = LogBinner(ComplexF64)

    # block of maximally correlated values:
    N_corr = 16

    # number of blocks
    N_blocks = 131_072 # 2^17

    for _ in 1:N_blocks
        x = rand(rng, ComplexF64)
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
        x = rand(rng, Float64, 3)
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
    @test (append!(Bf, rand(rng, 1:10, 10000)); true)
    @test (append!(Bc, rand(rng, 10000)); true)
end


@testset "Sum-type heuristic" begin
    # numbers
    @test typeof(LogBinner(zero(Int64))) == LogBinner{Float64, 32, BinningAnalysis.Variance{Float64}}
    @test typeof(LogBinner(zero(ComplexF16))) == LogBinner{ComplexF64, 32, BinningAnalysis.Variance{ComplexF64}}

    # arrays
    @test typeof(LogBinner(zeros(Int64, 2,2))) == LogBinner{Matrix{Float64}, 32, BinningAnalysis.Variance{Matrix{Float64}}}
    @test typeof(LogBinner(zeros(ComplexF16, 2,2))) == LogBinner{Matrix{ComplexF64}, 32, BinningAnalysis.Variance{Matrix{ComplexF64}}}
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
    push!(BA, rand(rng, ))
    @test BinningAnalysis._reliable_level(BA) == 1
    @test isnan(std_error(BA, BinningAnalysis._reliable_level(BA)))

    # Two elements should return some value
    push!(BA, rand(rng, ))
    @test BinningAnalysis._reliable_level(BA) == 1
    @test !isnan(std_error(BA, BinningAnalysis._reliable_level(BA)))

    # same behavior up to (including) 63 values (31 binned in first binned lvl)
    append!(BA, rand(rng, 61))
    @test BinningAnalysis._reliable_level(BA) == 1
    @test !isnan(std_error(BA, BinningAnalysis._reliable_level(BA)))

    # at 64 or more values, the lvl should be increasing
    push!(BA, rand(rng, ))
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
    @test readline(read_pipe) == "LogBinner{Float64,32}()"
    # full
    @test readline(read_pipe) == "LogBinner{Float64,32}"
    @test readline(read_pipe) == "| Count: 0"
    @test length(readlines(read_pipe)) == 0
    close(read_pipe);

    # filled binner
    StableRNGs.seed!(rng, 123)
    append!(B, rand(rng, 1000))
    (read_pipe, write_pipe) = redirect_stdout()
    show(write_pipe, MIME"text/plain"(), B)
    redirect_stdout(oldstdout);
    close(write_pipe);
    @test readline(read_pipe) == "LogBinner{Float64,32}"
    @test readline(read_pipe) == "| Count: 1000"
    @test readline(read_pipe) == "| Mean: 0.49387"
    @test readline(read_pipe) == "| StdError: 0.00952"
    @test length(readlines(read_pipe)) == 0
    close(read_pipe);
end
