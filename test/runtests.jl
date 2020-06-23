using BinningAnalysis
using Test, Statistics, Random


@testset "All Tests" begin

    @testset "Logarithmic Binning" begin
        include("logbinning.jl")
    end


    @testset "Full Binning" begin
        include("fullbinning.jl")
    end


    @testset "Error Propagator" begin
        include("errorpropagator.jl")
    end


    @testset "Generic functions" begin
        x = [0.0561823, 0.846613, 0.813439, 0.357134, 0.157445, 0.103298, 0.948842, 0.629425, 0.290206, 0.00695332, 0.869828, 0.949165, 0.897995, 0.916239, 0.457564, 0.349827, 0.398683, 0.264218, 0.72754, 0.934315, 0.666448, 0.134813, 0.364933, 0.829088, 0.256443, 0.595029, 0.172097, 0.241686, 0.489935, 0.239663, 0.391291, 0.00751015, 0.138935, 0.0569876, 0.571786, 0.694996, 0.798602, 0.923308, 0.73978, 0.414774, 0.835145, 0.731303, 0.271647, 0.707796, 0.00348624, 0.0905812, 0.316176, 0.921054, 0.131037, 0.599667, 0.805071, 0.440813, 0.086516, 0.363658, 0.476161, 0.931257, 0.28974, 0.78717, 0.60822, 0.144024, 0.214432, 0.061922, 0.626495, 0.512072, 0.758078, 0.840485, 0.242576, 0.147441, 0.599222, 0.993569, 0.0365044, 0.0983033, 0.713144, 0.422394, 0.480044, 0.968745, 0.518475, 0.431319, 0.4432, 0.526007, 0.612975, 0.468387, 0.262145, 0.888011, 0.105744, 0.325821, 0.769525, 0.289073, 0.336083, 0.443037, 0.489698, 0.141654, 0.915284, 0.319068, 0.341001, 0.704346, 0.0794996, 0.0412352, 0.70016, 0.0195158]
        @test std_error(x) ≈ std_error(x; method=:log)
        @test std_error(x; method=:log) ≈ 0.03384198827323159
        @test std_error(x; method=:full) ≈ 0.030761063167290947
        @test std_error(x; method=:jackknife) ≈ 0.029551526503551463
        @test tau(x) ≈ tau(x; method=:log)
        @test tau(x; method=:log) ≈ 0.15572524869061377
        @test tau(x; method=:full) ≈ 0.04176737474771308

        x = x .+ 1im
        @test std_error(x)≈ std_error(x; method=:log)
        @test std_error(x; method=:log) ≈ 0.0338419882732316
        @test std_error(x; method=:full) ≈ 0.03076106316729094
        @test std_error(x; method=:jackknife) ≈ 0.029551526503551445
        @test tau(x) ≈ tau(x; method=:log)
        @test tau(x; method=:log) ≈ 0.15572524869061377
        @test tau(x; method=:full) ≈ 0.04176737474771308


        x = Array{Float64,2}[[0.311954 0.476706 0.314101; 0.420978 0.478085 0.0194284], [0.491386 0.493583 0.295477; 0.136896 0.1634 0.84641], [0.318536 0.705903 0.121377; 0.764174 0.240484 0.25894], [0.286923 0.172458 0.392881; 0.124348 0.140628 0.730131], [0.287451 0.221914 0.382938; 0.29568 0.249575 0.87685]]
        @test std_error(x) ≈ std_error(x; method=:log)
        @test std_error(x, method=:log) ≈ [0.03856212859659066 0.09765048982846934 0.04879645809318544; 0.11744109653881814 0.05978167652935139 0.1722342286233907]
        @test std_error(x, method=:full) ≈ [0.03856212859659072 0.09765048982846936 0.048796458093185405; 0.11744109653881814 0.05978167652935137 0.17223422862339066]
        @test tau(x) ≈ tau(x; method=:log)
        @test tau(x; method=:log) ≈ [0.0 0.0 0.0; 0.0 0.0 0.0]
        @test isapprox(tau(x; method=:full), [0.0 -5.55112e-17 -5.55112e-17; -1.11022e-16 0.0 0.0], atol=1e-16)

        x = ["this", "should", "error"]
        @test_throws ErrorException std_error(x, method=:jackknife)
        @test_throws ArgumentError std_error(x, method=:whatever)
    end



    @testset "Jackknife" begin
        Random.seed!(123)
        g(x1, x2) = x1^2 - x2

        # Real
        ts = rand(Float64, 1000)
        ts2 = rand(Float64, 1000)

        @test isapprox.(
            jackknife_full(identity, ts),
            (0.5032466005013134, 1.1091128016005314e-13, 0.008897060002353763),
            atol = 1e-12
        ) |> all
        @test isapprox.(
            jackknife_full(identity, 1 ./ ts),
            (7.97707906607684, -5.323741447682551e-12, 2.522127468106169),
            atol = 1e-12
        ) |> all
        @test isapprox.(
            jackknife_full(g, ts, ts2.^2),
            (-0.05969375347674344, 7.915767673482427e-5, 0.013117185693279606),
            atol = 1e-12
        ) |> all

        # check consistency with Julia's Statistics
        @test jackknife(identity, ts)[2] ≈ std(ts)/sqrt(length(ts))

        # Complex
        ts = rand(ComplexF64, 1000)
        ts2 = rand(ComplexF64, 1000)

        @test isapprox.(
            jackknife_full(identity, ts),
            (0.5026931517594448 + 0.49854565281725627im, 4.4364512064021255e-13 - 3.327338404801594e-13im, 0.012946143510730868),
            atol = 1e-12
        ) |> all
        @test isapprox.(
            jackknife_full(identity, 1 ./ ts),
            (1.1496877962692906 - 1.0838709685240246im, 1.3309353619206377e-12 - 6.654676809603188e-13im, 0.05676635162628594),
            atol = 1e-12
        ) |> all
        @test isapprox.(
            jackknife_full(g, ts, ts2.^2),
            (0.02004836872405491 - 0.01825289014938747im, -3.1784715169222433e-6 + 5.949834723031899e-7im, 0.026446578773003968),
            atol = 1e-12
        ) |> all

        # check consistency with Julia's Statistics
        @test jackknife(identity, ts)[2] ≈ std(ts)/sqrt(length(ts))
    end
end
