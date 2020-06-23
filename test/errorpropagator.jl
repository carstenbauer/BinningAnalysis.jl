@testset "Constructors and basic properties" begin
    # numbers
    for T in (Float64, ComplexF64)
        ep = ErrorPropagator(T, N_args=1)

        @test length(ep) == 0
        @test ndims(ep) == 0
        @test isempty(ep)
        @test eltype(ep) == T
        @test capacity(ep) == 2^32 - 1
        @test BinningAnalysis.nlevels(ep) == 32

        append!(ep, rand(1000))
        @test length(ep) == 1000
        @test !isempty(ep)

        empty!(ep)
        @test length(ep) == 0
        @test isempty(ep)
    end

    # arrays
    for T in (Float64, ComplexF64)
        ep = ErrorPropagator(zeros(T, 2, 3))

        @test length(ep) == 0
        @test ndims(ep) == 2
        @test isempty(ep)
        @test eltype(ep) == Array{T, 2}
        @test capacity(ep) == 2^32 - 1
        @test BinningAnalysis.nlevels(ep) == 32

        append!(ep, [rand(T, 2,3) for _ in 1:1000])
        @test length(ep) == 1000
        @test !isempty(ep)
        empty!(ep)
        @test length(ep) == 0
        @test isempty(ep)
    end

    # Constructor arguments
    ep = ErrorPropagator(capacity=12345)
    @test capacity(ep) == 16383
    @test_throws ArgumentError ErrorPropagator(capacity=0)
    @test_throws ArgumentError ErrorPropagator(capacity=-1)


    # Test error on overflow
    ep = ErrorPropagator(N_args=1, capacity=1)
    push!(ep, 1.0)
    @test_throws OverflowError push!(ep, 2.0)

    ep = ErrorPropagator(zeros(2,2), capacity=1)
    push!(ep, rand(2,2))
    @test_throws OverflowError push!(ep, rand(2,2))

    # time series constructor (#26)
    x = rand(16)
    ep = ErrorPropagator(x)
    @test length(ep) == 16
    @test mean(ep, 1) == mean(x)
    @test all(m ≈ [mean(x)] for m in all_means(ep))

    x = [rand(2,3) for _ in 1:8]
    ep = ErrorPropagator(x)
    @test length(ep) == 8
    @test mean(ep, 1) == mean(x)
    @test all(m ≈ [mean(x)] for m in all_means(ep))
end



@testset "Check variance for Float64 values" begin
    # NOTE
    # Due to the different (mathematically equivalent) versions of the variance
    # calculated here, the values are onyl approximately the same. (Float error)
    Random.seed!(1234)
    xs = rand(Float64, 1_000_000)
    ys = rand(Float64, 1_000_000)
    ep = ErrorPropagator(Float64, N_args=2)

    # Test small set (off by one errors are large here)
    for (x, y) in zip(xs[1:10], ys[1:10]); push!(ep, x, y) end
    @test var(ep, 1, 1) ≈ var(xs[1:10])
    @test var(ep, 2, 1) ≈ var(ys[1:10])
    @test varN(ep, 1, 1) ≈ var(xs[1:10])/10
    @test varN(ep, 2, 1) ≈ var(ys[1:10])/10
    @test vars(ep, 1) ≈ [var(xs[1:10]), var(ys[1:10])]
    @test covmat(ep, 1) ≈ [
        cov(xs[1:10], xs[1:10]) cov(xs[1:10], ys[1:10]);
        cov(ys[1:10], xs[1:10]) cov(ys[1:10], ys[1:10])
    ]

    # Test full set
    for (x, y) in zip(xs[11:end], ys[11:end]); push!(ep, x, y) end
    @test var(ep, 1, 1) ≈ var(xs)
    @test var(ep, 2, 1) ≈ var(ys)
    @test varN(ep, 1, 1) ≈ var(xs)/1_000_000
    @test varN(ep, 2, 1) ≈ var(ys)/1_000_000
    @test vars(ep, 1) ≈ [var(xs), var(ys)]
    @test covmat(ep, 1) ≈ [
        cov(xs, xs) cov(xs, ys);
        cov(ys, xs) cov(ys, ys)
    ]

    # all_* methods
    @test isapprox(first.(all_vars(ep)), [0.0833563, 0.0417223, 0.0208426, 0.0104261, 0.00525425, 0.00263066, 0.00129848, 0.000643458, 0.000324319, 0.000152849, 7.28893e-5, 3.70618e-5, 1.6847e-5, 8.35098e-6, 4.03675e-6, 2.06069e-6, 9.23046e-7, 5.40964e-7, 7.6288e-7], atol=1e-6)
    @test isapprox(first.(all_varNs(ep)), zero(first.(all_varNs(ep))), atol=1e-6)
    @test isapprox(first.(all_taus(ep)), [0.0, 0.000530439, 8.45695e-5, 0.000316855, 0.00426884, 0.0049475, -0.00152239, -0.00592876, -0.00195134, -0.0305476, -0.0520332, -0.0444472, -0.0858441, -0.0894091, -0.103052, -0.0879748, -0.130883, -0.0364442, 1.02534], atol=1e-6)
    @test isapprox(first.(all_std_errors(ep)), [0.000288715, 0.000288868, 0.000288739, 0.000288806, 0.000289945, 0.00029014, 0.000288275, 0.000286998, 0.000288151, 0.000279756, 0.000273279, 0.000275584, 0.000262764, 0.000261631, 0.000257247, 0.000262087, 0.000248065, 0.000277994, 0.000504275], atol=1e-6)

    @test isapprox(tau(ep, 1), -0.10305205108202309)
    @test isapprox(std_error(ep, 1), 0.00025724734978688446)
end



@testset "Check variance for Float64 vectors" begin
    Random.seed!(1234)
    xs = [rand(Float64, 3) for _ in 1:1_000_000]
    ys = [rand(Float64, 3) for _ in 1:1_000_000]
    ep = ErrorPropagator(zeros(Float64, 3), zeros(Float64, 3))

    # Test small set (off by one errors are large here)
    for (x, y) in zip(xs[1:10], ys[1:10]); push!(ep, x, y) end
    @test var(ep, 1, 1) ≈ var(xs[1:10])
    @test var(ep, 2, 1) ≈ var(ys[1:10])
    @test varN(ep, 1, 1) ≈ var(xs[1:10])/10
    @test varN(ep, 2, 1) ≈ var(ys[1:10])/10
    @test vars(ep, 1) ≈ [var(xs[1:10]), var(ys[1:10])]
    @test covmat(ep, 1) ≈ reshape([
        [cov([xs[i][j] for i in 1:10], [xs[i][j] for i in 1:10]) for j in 1:3],
        [cov([ys[i][j] for i in 1:10], [xs[i][j] for i in 1:10]) for j in 1:3],
        [cov([xs[i][j] for i in 1:10], [ys[i][j] for i in 1:10]) for j in 1:3],
        [cov([ys[i][j] for i in 1:10], [ys[i][j] for i in 1:10]) for j in 1:3]
    ], (2, 2))

    # Test full set
    for (x, y) in zip(xs[11:end], ys[11:end]); push!(ep, x, y) end
    @test var(ep, 1, 1) ≈ var(xs)
    @test var(ep, 2, 1) ≈ var(ys)
    @test varN(ep, 1, 1) ≈ var(xs)/1_000_000
    @test varN(ep, 2, 1) ≈ var(ys)/1_000_000
    @test vars(ep, 1) ≈ [var(xs), var(ys)]
    @test covmat(ep, 1) ≈ reshape([
        [cov([xs[i][j] for i in 1:1_000_000], [xs[i][j] for i in 1:1_000_000]) for j in 1:3],
        [cov([ys[i][j] for i in 1:1_000_000], [xs[i][j] for i in 1:1_000_000]) for j in 1:3],
        [cov([xs[i][j] for i in 1:1_000_000], [ys[i][j] for i in 1:1_000_000]) for j in 1:3],
        [cov([ys[i][j] for i in 1:1_000_000], [ys[i][j] for i in 1:1_000_000]) for j in 1:3]
    ], (2, 2))

    # all_std_errors for <:AbstractArray
    @test all(isapprox.(first.(all_std_errors(ep)), Ref(zeros(3)), atol=1e-2))

    @test all(isapprox.(tau(ep, 1), [0.0313041, -0.153329, -0.0566459], atol=1e-6))
    @test all(isapprox.(std_error(ep, 1), [0.000297531, 0.000240407, 0.000271932], atol=1e-6))
end



@testset "Check variance for complex values" begin
    # NOTE
    # Due to the different (mathematically equivalent) versions of the variance
    # calculated here, the values are onyl approximately the same. (Float error)
    Random.seed!(1234)
    xs = rand(ComplexF64, 1_000_000)
    ys = rand(ComplexF64, 1_000_000)
    ep = ErrorPropagator(ComplexF64, N_args=2)

    # Test small set (off by one errors are large here)
    for (x, y) in zip(xs[1:10], ys[1:10]); push!(ep, x, y) end
    @test var(ep, 1, 1) ≈ var(xs[1:10])
    @test var(ep, 2, 1) ≈ var(ys[1:10])
    @test varN(ep, 1, 1) ≈ var(xs[1:10])/10
    @test varN(ep, 2, 1) ≈ var(ys[1:10])/10
    @test vars(ep, 1) ≈ [var(xs[1:10]), var(ys[1:10])]
    @test covmat(ep, 1) ≈ [
        cov(xs[1:10], xs[1:10]) cov(xs[1:10], ys[1:10]);
        cov(ys[1:10], xs[1:10]) cov(ys[1:10], ys[1:10])
    ]

    # Test full set
    for (x, y) in zip(xs[11:end], ys[11:end]); push!(ep, x, y) end
    @test var(ep, 1, 1) ≈ var(xs)
    @test var(ep, 2, 1) ≈ var(ys)
    @test varN(ep, 1, 1) ≈ var(xs)/1_000_000
    @test varN(ep, 2, 1) ≈ var(ys)/1_000_000
    @test vars(ep, 1) ≈ [var(xs), var(ys)]
    @test covmat(ep, 1) ≈ [
        cov(xs, xs) cov(xs, ys);
        cov(ys, xs) cov(ys, ys)
    ]

    # all_* methods
    @test isapprox(first.(all_vars(ep)), [0.16671474067121222, 0.08324845751233179, 0.041527133392489035, 0.020847602123934883, 0.010430741538377142, 0.005111097271805531, 0.0025590988213273214, 0.001283239131297187, 0.0006322480081128456, 0.0003060164750540162, 0.00015750782337442537, 8.006142368921498e-5, 3.810111634139357e-5, 1.80535512880331e-5, 1.0002438211476061e-5, 5.505102193326117e-6, 2.8788397929968568e-6, 1.7242475507384114e-6, 7.900888818745955e-7])
    @test isapprox(first.(all_varNs(ep)), zero(first.(all_varNs(ep))), atol=1e-6)
    @test isapprox(first.(all_taus(ep)), [0.0, -0.00065328850247931, -0.0018180968845809553, 0.00019817179932868356, 0.0005312186016332987, -0.009476150581268994, -0.008794711536776634, -0.007346737569564443, -0.014542478848703244, -0.030064159934323653, -0.01599670814224563, -0.007961042363178128, -0.03167873601168558, -0.056188229083248886, -0.008218660661725774, 0.05035147373711113, 0.0756019296606737, 0.2387501479629205, 0.289861051172009])
    @test isapprox(first.(all_std_errors(ep)), [0.0004083071646092097, 0.0004080403350462593, 0.00040756414657076514, 0.0004083880715587553, 0.0004085240073900606, 0.00040441947616030687, 0.00040470028980091994, 0.0004052963382191276, 0.00040232555162611277, 0.00039584146247874917, 0.00040172249945971054, 0.00040504357104527986, 0.00039516087376314515, 0.0003846815937765092, 0.00040493752222959487, 0.0004283729758565588, 0.00043808977717638777, 0.0004963074437049236, 0.0005131890106236348])

    @test isapprox(tau(ep, 1), -0.008218660661725774)
    @test isapprox(std_error(ep, 1), 0.00040493752222959487)
end



@testset "Check variance for complex vectors" begin
    Random.seed!(1234)
    xs = [rand(ComplexF64, 3) for _ in 1:1_000_000]
    ys = [rand(ComplexF64, 3) for _ in 1:1_000_000]
    ep = ErrorPropagator(zeros(ComplexF64, 3), zeros(ComplexF64, 3))

    # Test small set (off by one errors are large here)
    for (x, y) in zip(xs[1:10], ys[1:10]); push!(ep, x, y) end
    @test var(ep, 1, 1) ≈ var(xs[1:10])
    @test var(ep, 2, 1) ≈ var(ys[1:10])
    @test varN(ep, 1, 1) ≈ var(xs[1:10])/10
    @test varN(ep, 2, 1) ≈ var(ys[1:10])/10
    @test vars(ep, 1) ≈ [var(xs[1:10]), var(ys[1:10])]
    @test covmat(ep, 1) ≈ reshape([
        [cov([xs[i][j] for i in 1:10], [xs[i][j] for i in 1:10]) for j in 1:3],
        [cov([ys[i][j] for i in 1:10], [xs[i][j] for i in 1:10]) for j in 1:3],
        [cov([xs[i][j] for i in 1:10], [ys[i][j] for i in 1:10]) for j in 1:3],
        [cov([ys[i][j] for i in 1:10], [ys[i][j] for i in 1:10]) for j in 1:3]
    ], (2, 2))

    # Test full set
    for (x, y) in zip(xs[11:end], ys[11:end]); push!(ep, x, y) end
    @test var(ep, 1, 1) ≈ var(xs)
    @test var(ep, 2, 1) ≈ var(ys)
    @test varN(ep, 1, 1) ≈ var(xs)/1_000_000
    @test varN(ep, 2, 1) ≈ var(ys)/1_000_000
    @test vars(ep, 1) ≈ [var(xs), var(ys)]
    @test covmat(ep, 1) ≈ reshape([
        [cov([xs[i][j] for i in 1:1_000_000], [xs[i][j] for i in 1:1_000_000]) for j in 1:3],
        [cov([ys[i][j] for i in 1:1_000_000], [xs[i][j] for i in 1:1_000_000]) for j in 1:3],
        [cov([xs[i][j] for i in 1:1_000_000], [ys[i][j] for i in 1:1_000_000]) for j in 1:3],
        [cov([ys[i][j] for i in 1:1_000_000], [ys[i][j] for i in 1:1_000_000]) for j in 1:3]
    ], (2, 2))

    # all_std_errors for <:AbstractArray
    @test all(isapprox.(first.(all_std_errors(ep)), Ref(zeros(3)), atol=1e-2))

    @test all(isapprox.(tau(ep, 1), [-0.101203, -0.0831874, -0.0112827], atol=1e-6))
    @test all(isapprox.(std_error(ep, 1), [0.000364498, 0.00037268, 0.000403603], atol=1e-6))
end



@testset "Type promotion" begin
    epf = ErrorPropagator(zero(1.)) # Float64 ErrorPropagator
    epc = ErrorPropagator(zero(im)) # Float64 ErrorPropagator

    # Check that this doesn't throw (TODO: is there a better way?)
    @test (append!(epf, rand(1:10, 10000)); true)
    @test (append!(epc, rand(10000)); true)
end



@testset "Sum-type heuristic" begin
    # numbers
    @test typeof(ErrorPropagator(zero(Int64))) == ErrorPropagator{Float64, 32}
    @test typeof(ErrorPropagator(zero(ComplexF16))) == ErrorPropagator{ComplexF64, 32}

    # arrays
    @test typeof(ErrorPropagator(zeros(Int64, 2,2))) == ErrorPropagator{Matrix{Float64}, 32}
    @test typeof(ErrorPropagator(zeros(ComplexF16, 2,2))) == ErrorPropagator{Matrix{ComplexF64}, 32}
end



@testset "Indexing Bounds" begin
    ep = ErrorPropagator(zero(Float64); capacity=1)
    for func in [:var, :varN, :mean, :tau]
        # check if func(ep, 0) throws BoundsError
        # It should as level 1 is now the initial level
        @test_throws BoundsError @eval $func($ep, 1, 0)
        # Check that level 1 exists
        @test (@eval $func($ep, 1, 1); true)
        # Check that level 2 throws BoundsError
        @test_throws BoundsError @eval $func($ep, 1, 2)
    end
end



@testset "_reliable_level" begin
    ep = ErrorPropagator(N_args=1)
    # Empty Binner
    @test BinningAnalysis._reliable_level(ep) == 1
    @test isnan(std_error(ep, 1, BinningAnalysis._reliable_level(ep)))

    # One Element should still return NaN (due to 1/(n-1))
    push!(ep, rand())
    @test BinningAnalysis._reliable_level(ep) == 1
    @test isnan(std_error(ep, 1, BinningAnalysis._reliable_level(ep)))

    # Two elements should return some value
    push!(ep, rand())
    @test BinningAnalysis._reliable_level(ep) == 1
    @test !isnan(std_error(ep, 1, BinningAnalysis._reliable_level(ep)))

    # same behavior up to (including) 63 values (31 binned in first binned lvl)
    append!(ep, rand(61))
    @test BinningAnalysis._reliable_level(ep) == 1
    @test !isnan(std_error(ep, 1, BinningAnalysis._reliable_level(ep)))

    # at 64 or more values, the lvl should be increasing
    push!(ep, rand())
    @test BinningAnalysis._reliable_level(ep) == 2
    @test !isnan(std_error(ep, 1, BinningAnalysis._reliable_level(ep)))
end



@testset "Cosmetics (show, print, etc.)" begin
    ep = ErrorPropagator(N_args=1);
    # empty binner
    io = IOBuffer()
    println(io, ep) # compact
    show(io, MIME"text/plain"(), ep) # full


    # compact
    l = String(take!(io))
    @test l == "ErrorPropagator{Float64,32}()\nErrorPropagator{Float64,32}\n| Count: 0"
    @test length(readlines(io)) == 0


    # filled binner
    Random.seed!(1234)
    append!(ep, rand(1000))
    show(io, MIME"text/plain"(), ep)

    l = String(take!(io))
    @test l == "ErrorPropagator{Float64,32}\n| Count: 1000\n| Means: [0.49685]\n| StdErrors: [0.00733]"
    @test length(readlines(io)) == 0
    close(io);
end



@testset "Error Propagation" begin
    # These tests are taken from Jackknife.jl
    Random.seed!(123)

    g(x1, x2) = x1^2 - x2
    g(v) = v[1]^2 - v[2]
    grad_g(x1, x2) = [2x1, -1.0]
    grad_g(v) = [2v[1], -1.0]

    # derivative of identity is x -> 1
    grad_identity(v) = 1

    # Real
    # comparing to results from mean(ts), std(ts)/sqrt(10) (etc)
    ts = rand(Float64, 1000)
    ts2 = rand(Float64, 1000)

    ep = ErrorPropagator(ts)
    @test isapprox(mean(ep, 1), 0.5032466005013134, atol=1e-12)
    @test isapprox(std_error(ep, grad_identity, 1), 0.008897060002353763, atol=1e-12)
    # check consistency with Julia's Statistics
    @test std_error(ep, grad_identity, 1) ≈ std(ts)/sqrt(length(ts))

    ep = ErrorPropagator(1 ./ ts)
    @test isapprox(mean(ep, 1), 7.977079066071383, atol=1e-12)
    @test isapprox(std_error(ep, grad_identity, 1), 2.5221274681061696, atol=1e-12)

    ep = ErrorPropagator(ts, ts2.^2)
    @test isapprox(mean(ep, g), -0.05969375347674344, atol=1e-3)
    @test isapprox(std_error(ep, grad_g, 1), 0.013117185693279606, atol=1e-4)


    # Complex
    ts = rand(ComplexF64, 1000)
    ts2 = rand(ComplexF64, 1000)

    ep = ErrorPropagator(ts)
    @test isapprox(mean(ep, 1), 0.5026931517594448 + 0.49854565281725627im, atol=1e-12)
    @test isapprox(std_error(ep, grad_identity, 1), 0.012946143510730868, atol=1e-12)
    # check consistency with Julia's Statistics
    @test std_error(ep, grad_identity, 1) ≈ std(ts)/sqrt(length(ts))

    ep = ErrorPropagator(1 ./ ts)
    @test isapprox(mean(ep, 1), 1.1496877962692906 - 1.0838709685240246im, atol=1e-11)
    @test isapprox(std_error(ep, grad_identity, 1), 0.05676635162628594, atol=1e-12)

    ep = ErrorPropagator(ts, ts2.^2)
    @test isapprox(mean(ep, g), 0.02004836872405491 - 0.01825289014938747im, atol=1e-4)
    @test isapprox(std_error(ep, grad_g, 1), 0.026446578773003968, atol=1e-5)
end
