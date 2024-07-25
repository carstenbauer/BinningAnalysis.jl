@testset "Constructors and basic properties" begin
    # numbers
    for T in (Float64, ComplexF64)
        ep = ErrorPropagator(T, N_args=1)
        ep2 = ErrorPropagator(T, N_args=1)

        @test length(ep) == 0
        @test ndims(ep) == 0
        @test isempty(ep)
        @test eltype(ep) == T
        @test capacity(ep) == 2^32 - 1
        @test BinningAnalysis.nlevels(ep) == 32
        @test ep == ep2
        @test ep ≈ ep2

        data = rand(rng, 1000)
        append!(ep, data)
        @test length(ep) == 1000
        @test !isempty(ep)
        @test ep != ep2
        @test !(ep ≈ ep2)
        append!(ep2, data)
        @test ep == ep2
        @test ep ≈ ep2

        empty!(ep)
        @test length(ep) == 0
        @test isempty(ep)
    end

    # arrays
    for T in (Float64, ComplexF64)
        ep = ErrorPropagator(zeros(T, 2, 3))
        ep2 = ErrorPropagator(zeros(T, 2, 3))

        @test length(ep) == 0
        @test ndims(ep) == 2
        @test isempty(ep)
        @test eltype(ep) == Array{T, 2}
        @test capacity(ep) == 2^32 - 1
        @test BinningAnalysis.nlevels(ep) == 32
        @test ep == ep2
        @test ep ≈ ep2

        data = [rand(rng, T, 2,3) for _ in 1:1000]
        append!(ep, data)
        @test length(ep) == 1000
        @test !isempty(ep)
        @test ep != ep2
        @test !(ep ≈ ep2)
        append!(ep2, data)
        @test ep == ep2
        @test ep ≈ ep2

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
    push!(ep, rand(rng, 2,2))
    @test_throws OverflowError push!(ep, rand(rng, 2,2))

    # time series constructor (#26)
    x = rand(rng, 16)
    ep = ErrorPropagator(x)
    @test length(ep) == 16
    @test mean(ep, 1) == mean(x)
    @test all(m ≈ [mean(x)] for m in all_means(ep))

    x = [rand(rng, 2,3) for _ in 1:8]
    ep = ErrorPropagator(x)
    @test length(ep) == 8
    @test mean(ep, 1) == mean(x)
    @test all(m ≈ [mean(x)] for m in all_means(ep))
end



@testset "Check variance for Float64 values" begin
    # NOTE
    # Due to the different (mathematically equivalent) versions of the variance
    # calculated here, the values are only approximately the same. (Float error)
    StableRNGs.seed!(rng, 123)
    xs = rand(rng, Float64, 1_000_000)
    ys = rand(rng, Float64, 1_000_000)
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
    @test isapprox(first.(all_vars(ep)), [0.08325406274030639, 0.041664887985400856, 0.02086378726697563, 0.010404153111327696, 0.005186509467804135, 0.002587078870339432, 0.001287770691588086, 0.0006406912824759581, 0.00031618954047712977, 0.00015796947195956257, 7.688354009283138e-5, 3.805137200274533e-5, 1.7766726465417992e-5, 8.513139774846135e-6, 3.718644130956683e-6, 2.0176056744203308e-6, 1.198040510186349e-6, 4.1842538345271407e-7, 4.374011807750655e-7], atol=1e-6)
    @test isapprox(first.(all_varNs(ep)), zero(first.(all_varNs(ep))), atol=1e-6)
    @test isapprox(first.(all_taus(ep)), [0.0, 0.0004547119263806909, 0.0012076667550950937, -0.00012514614301650795, -0.0016210095132663804, -0.0028079043475808807, -0.005025211089571158, -0.007449064127587446, -0.01383932219829298, -0.014224629670230415, -0.02690525630387247, -0.03170980256141187, -0.06269730070559826, -0.08092234827197647, -0.1338834575186591, -0.0960946513177322, -0.020327508851475773, -0.1410081813940125, 0.37563530711097126], atol=1e-6)
    @test isapprox(first.(all_std_errors(ep)), [0.00028853780123288246, 0.00028866897299640936, 0.0002888860485864669, 0.00028850168958018525, 0.0002880696990050605, 0.0002877264740180539, 0.00028708417626479785, 0.0002863803991823409, 0.0002845166131232874, 0.000284403843712177, 0.00028066727445067965, 0.0002792384696929228, 0.00026984153262076416, 0.0002641587292141796, 0.0002469035827929826, 0.00025933284111480696, 0.00028261169003733125, 0.00024448937559525804, 0.00038183817723178037], atol=1e-6)

    @test isapprox(tau(ep, 1), -0.1338834575186591)
    @test isapprox(std_error(ep, 1), 0.0002469035827929826)
end



@testset "Check variance for Float64 vectors" begin
    StableRNGs.seed!(rng, 123)
    xs = [rand(rng, Float64, 3) for _ in 1:1_000_000]
    ys = [rand(rng, Float64, 3) for _ in 1:1_000_000]
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

    @test all(isapprox.(tau(ep, 1), [-0.06939313732460628, -0.05220987781564013, 0.13728303267264075], atol=1e-6))
    @test all(isapprox.(std_error(ep, 1), [0.0002677468110272373, 0.00027324026280709543, 0.00032601860649145024], atol=1e-6))
end



@testset "Check variance for complex values" begin
    # NOTE
    # Due to the different (mathematically equivalent) versions of the variance
    # calculated here, the values are onyl approximately the same. (Float error)
    StableRNGs.seed!(rng, 123)
    xs = rand(rng, ComplexF64, 1_000_000)
    ys = rand(rng, ComplexF64, 1_000_000)
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
    @test isapprox(first.(all_vars(ep)), [0.16658475406392415, 0.08321852969182086, 0.04163861706031047, 0.020801210327666575, 0.010404232627790222, 0.005208087488829505, 0.002589371281786268, 0.0012979310682935674, 0.0006498645790018465, 0.00033049882070657066, 0.00016135937775918485, 8.244734583795488e-5, 4.190438084472614e-5, 2.1161425558080538e-5, 1.0185807653328993e-5, 5.794287254556885e-6, 2.412065927770435e-6, 2.1323503403003485e-6, 2.657209370315883e-7])
    @test isapprox(first.(all_varNs(ep)), zero(first.(all_varNs(ep))), atol=1e-6)
    @test isapprox(first.(all_taus(ep)), [0.0, -0.00044330191292812904, -9.090214423418397e-5, -0.0005254725847371189, -0.0003512686978415225, 0.0002222459643324015, -0.002596252083402084, -0.0013174794805482781, -0.0006266293031622627, 0.00792831442226083, -0.0037744232968683344, 0.007097663665263987, 0.015471138766913417, 0.02061879500256958, 0.0011876815245235317, 0.07971364057455732, -0.017349603544782877, 0.41431379449307415, -0.23414840339108273])
    @test isapprox(first.(all_std_errors(ep)), [0.0004081479560942626, 0.00040796698320285887, 0.0004081108528834315, 0.00040793342915398906, 0.0004080045613037231, 0.00040823865525271385, 0.0004070869219642424, 0.0004076098748477781, 0.00040789211844213303, 0.0004113711544095814, 0.0004066045146215672, 0.0004110346447395122, 0.0004144143648054135, 0.00041647846025109046, 0.0004086324183581892, 0.00043948027087143456, 0.0004010042333750302, 0.0005519252460123237, 0.00029761324849071505])

    @test isapprox(tau(ep, 1), 0.0011876815245235317)
    @test isapprox(std_error(ep, 1), 0.0004086324183581892)
end



@testset "Check variance for complex vectors" begin
    StableRNGs.seed!(rng, 123)
    xs = [rand(rng, ComplexF64, 3) for _ in 1:1_000_000]
    ys = [rand(rng, ComplexF64, 3) for _ in 1:1_000_000]
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

    @test all(isapprox.(tau(ep, 1), [0.02509990727281808, 0.0756740457843339, 0.021586917851548426], atol=1e-6))
    @test all(isapprox.(std_error(ep, 1), [0.00041845464561767837, 0.0004382193124492061, 0.00041696433722574546], atol=1e-6))
end



@testset "Type promotion" begin
    epf = ErrorPropagator(zero(1.)) # Float64 ErrorPropagator
    epc = ErrorPropagator(zero(im)) # Float64 ErrorPropagator

    # Check that this doesn't throw (TODO: is there a better way?)
    @test (append!(epf, rand(rng, 1:10, 10000)); true)
    @test (append!(epc, rand(rng, 10000)); true)
end



@testset "Sum-type heuristic" begin
    # numbers
    @test typeof(ErrorPropagator(zero(Int))) == ErrorPropagator{Float64, 32}
    @test typeof(ErrorPropagator(zero(ComplexF16))) == ErrorPropagator{ComplexF64, 32}

    # arrays
    @test typeof(ErrorPropagator(zeros(Int, 2,2))) == ErrorPropagator{Matrix{Float64}, 32}
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
    push!(ep, rand(rng, ))
    @test BinningAnalysis._reliable_level(ep) == 1
    @test isnan(std_error(ep, 1, BinningAnalysis._reliable_level(ep)))

    # Two elements should return some value
    push!(ep, rand(rng, ))
    @test BinningAnalysis._reliable_level(ep) == 1
    @test !isnan(std_error(ep, 1, BinningAnalysis._reliable_level(ep)))

    # same behavior up to (including) 63 values (31 binned in first binned lvl)
    append!(ep, rand(rng, 61))
    @test BinningAnalysis._reliable_level(ep) == 1
    @test !isnan(std_error(ep, 1, BinningAnalysis._reliable_level(ep)))

    # at 64 or more values, the lvl should be increasing
    push!(ep, rand(rng, ))
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
    StableRNGs.seed!(rng, 123)
    append!(ep, rand(rng, 1000))
    show(io, MIME"text/plain"(), ep)

    l = String(take!(io))
    @test l == "ErrorPropagator{Float64,32}\n| Count: 1000\n| Means: [0.49413]\n| StdErrors: [0.00952]"
    @test length(readlines(io)) == 0
    close(io);
end



@testset "Error Propagation" begin
    # These tests are taken from Jackknife.jl
    StableRNGs.seed!(rng, 123)

    g(x1, x2) = x1^2 - x2
    g(v) = v[1]^2 - v[2]
    grad_g(x1, x2) = [2x1, -1.0]
    grad_g(v) = [2v[1], -1.0]

    # derivative of identity is x -> 1
    grad_identity(v) = 1

    # Real
    # comparing to results from mean(ts), std(ts)/sqrt(10) (etc)
    ts = rand(rng, Float64, 1000)
    ts2 = rand(rng, Float64, 1000)

    ep = ErrorPropagator(ts)
    @test isapprox(mean(ep, 1), 0.4938701707368938, atol=1e-12)
    @test isapprox(varN(ep, grad_identity, 1), 8.54933681103257e-5, atol=1e-12)
    @test isapprox(std_error(ep, grad_identity, 1), 0.009246262385976601, atol=1e-12)
    # check consistency with Julia's Statistics
    @test varN(ep, grad_identity, 1) ≈ std(ts)^2.0/length(ts)
    @test std_error(ep, grad_identity, 1) ≈ std(ts)/sqrt(length(ts))

    ep = ErrorPropagator(1 ./ ts)
    @test isapprox(mean(ep, 1), 8.100918442539456, atol=1e-12)
    @test isapprox(varN(ep, grad_identity, 1), 1.2769418115479338, atol=1e-12)
    @test isapprox(std_error(ep, grad_identity, 1), 1.1300185005334797, atol=1e-12)

    ep = ErrorPropagator(ts, ts2.^2)
    @test isapprox(mean(ep, g), -0.0932215528431069, atol=1e-3)
    @test isapprox(varN(ep, grad_g, 1), 0.00017206056051197917, atol=1e-4)
    @test isapprox(std_error(ep, grad_g, 1), 0.013117185693279606, atol=1e-4)


    # Complex
    ts = rand(rng, ComplexF64, 1000)
    ts2 = rand(rng, ComplexF64, 1000)

    ep = ErrorPropagator(ts)
    @test isapprox(mean(ep, 1), 0.5179888756352957 + 0.4818028227938359im, atol=1e-12)
    @test isapprox(varN(ep, grad_identity, 1), 0.00016572004644474712, atol=1e-12)
    @test isapprox(std_error(ep, grad_identity, 1), 0.012873229837330922, atol=1e-12)
    # check consistency with Julia's Statistics
    @test varN(ep, grad_identity, 1) ≈ std(ts)^2.0/length(ts)
    @test std_error(ep, grad_identity, 1) ≈ std(ts)/sqrt(length(ts))

    ep = ErrorPropagator(1 ./ ts)
    @test isapprox(mean(ep, 1), 1.1073219254991422 - 1.0192129144004423im, atol=1e-11)
    @test isapprox(varN(ep, grad_identity, 1), 0.0020457719406840333, atol=1e-12)
    @test isapprox(std_error(ep, grad_identity, 1), 0.04523021048684201, atol=1e-12)

    ep = ErrorPropagator(ts, ts2.^2)
    @test isapprox(mean(ep, g), 0.01459184115054761 + 0.015352631230090064im, atol=1e-4)
    @test isapprox(varN(ep, grad_g, 1), 0.0007026624982701395, atol=1e-5)
    @test isapprox(std_error(ep, grad_g, 1), 0.026538016921358373, atol=1e-5)
end
