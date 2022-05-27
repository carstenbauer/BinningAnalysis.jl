using BinningAnalysis
using Test, Statistics, StableRNGs

const rng = StableRNG(123)


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

    @testset "Incremental Binning" begin
        include("incrementbinner.jl")
    end

    include("systematic_correlation.jl")

    @testset "Generic functions" begin
        x = [0.0561823, 0.846613, 0.813439, 0.357134, 0.157445, 0.103298, 0.948842, 0.629425, 0.290206, 0.00695332, 0.869828, 0.949165, 0.897995, 0.916239, 0.457564, 0.349827, 0.398683, 0.264218, 0.72754, 0.934315, 0.666448, 0.134813, 0.364933, 0.829088, 0.256443, 0.595029, 0.172097, 0.241686, 0.489935, 0.239663, 0.391291, 0.00751015, 0.138935, 0.0569876, 0.571786, 0.694996, 0.798602, 0.923308, 0.73978, 0.414774, 0.835145, 0.731303, 0.271647, 0.707796, 0.00348624, 0.0905812, 0.316176, 0.921054, 0.131037, 0.599667, 0.805071, 0.440813, 0.086516, 0.363658, 0.476161, 0.931257, 0.28974, 0.78717, 0.60822, 0.144024, 0.214432, 0.061922, 0.626495, 0.512072, 0.758078, 0.840485, 0.242576, 0.147441, 0.599222, 0.993569, 0.0365044, 0.0983033, 0.713144, 0.422394, 0.480044, 0.968745, 0.518475, 0.431319, 0.4432, 0.526007, 0.612975, 0.468387, 0.262145, 0.888011, 0.105744, 0.325821, 0.769525, 0.289073, 0.336083, 0.443037, 0.489698, 0.141654, 0.915284, 0.319068, 0.341001, 0.704346, 0.0794996, 0.0412352, 0.70016, 0.0195158]
        @test std_error(x) ≈ std_error(x; method=:log)
        @test std_error(x; method=:log) ≈ 0.03384198827323159
        @test std_error(x; method=:full) ≈ 0.028771335389461323
        @test std_error(x; method=:jackknife) ≈ 0.029551526503551463
        @test std_error(x; method=:error_propagator) ≈ 0.03384198827323159
        @test tau(x) ≈ tau(x; method=:log)
        @test tau(x; method=:log) ≈ 0.15572524869061377
        @test tau(x; method=:full) ≈ -0.026052535205771943

        x = x .+ 1im
        @test std_error(x)≈ std_error(x; method=:log)
        @test std_error(x; method=:log) ≈ 0.0338419882732316
        @test std_error(x; method=:full) ≈ 0.028771335389461323
        @test std_error(x; method=:jackknife) ≈ 0.029551526503551445
        @test std_error(x; method=:error_propagator) ≈ 0.03384198827323173
        @test tau(x) ≈ tau(x; method=:log)
        @test tau(x; method=:log) ≈ 0.15572524869061377
        @test tau(x; method=:full) ≈ -0.026052535205771943


        x = Array{Float64,2}[[0.311954 0.476706 0.314101; 0.420978 0.478085 0.0194284], [0.491386 0.493583 0.295477; 0.136896 0.1634 0.84641], [0.318536 0.705903 0.121377; 0.764174 0.240484 0.25894], [0.286923 0.172458 0.392881; 0.124348 0.140628 0.730131], [0.287451 0.221914 0.382938; 0.29568 0.249575 0.87685]]
        @test std_error(x) ≈ std_error(x; method=:log)
        @test std_error(x, method=:log) ≈ [0.03856212859659066 0.09765048982846934 0.04879645809318544; 0.11744109653881814 0.05978167652935139 0.1722342286233907]
        @test std_error(x, method=:full) ≈ [0.03856212859659072 0.09765048982846936 0.048796458093185405; 0.11744109653881814 0.05978167652935137 0.17223422862339066]
        @test std_error(x, method=:error_propagator) ≈ [0.03856212859659066 0.09765048982846934 0.04879645809318544; 0.11744109653881814 0.05978167652935139 0.1722342286233907]
        @test tau(x) ≈ tau(x; method=:log)
        @test tau(x; method=:log) ≈ [0.0 0.0 0.0; 0.0 0.0 0.0]
        @test tau(x; method=:full) ≈ [0.0 0.0 0.0; 0.0 0.0 0.0]

        x = ["this", "should", "error"]
        @test_throws ErrorException std_error(x, method=:jackknife)
        @test_throws ArgumentError std_error(x, method=:whatever)
    end



    @testset "Jackknife" begin
        StableRNGs.seed!(rng, 123)
        g(x1, x2) = x1^2 - x2

        # Real
        ts = rand(rng, Float64, 1000)
        ts2 = rand(rng, Float64, 1000)

        @test isapprox.(
            jackknife_full(identity, ts),
            (0.49387017073689776, 0.0, 0.00924626238597665),
            atol = 1e-12
        ) |> all
        @test isapprox.(
            jackknife_full(identity, 1 ./ ts),
            (8.100918442541115, -1.7745804825608502e-12, 1.1300185005334797),
            atol = 1e-12
        ) |> all
        @test isapprox.(
            jackknife_full(g, ts, ts2.^2),
            (-0.0933070462112795, 8.549336816225661e-5, 0.013044632345917254),
            atol = 1e-12
        ) |> all

        # check consistency with Julia's Statistics
        @test jackknife(identity, ts)[2] ≈ std(ts)/sqrt(length(ts))

        # Complex
        ts = rand(rng, ComplexF64, 1000)
        ts2 = rand(rng, ComplexF64, 1000)

        @test isapprox.(
            jackknife_full(identity, ts),
            (0.5179888756356377 + 0.48180282279395215im, -3.327338404801594e-13 - 1.1091128016005314e-13im, 0.01287322983733096),
            atol = 1e-12
        ) |> all
        
        @test isapprox.(
            jackknife_full(identity, 1 ./ ts),
            (1.1073219254997184 - 1.0192129143995317im, -4.4364512064021255e-13 - 8.872902412804251e-13im, 0.04523021048684197),
            atol = 1e-12
        ) |> all
        @test isapprox.(
            jackknife_full(g, ts, ts2.^2),
            (0.014591533164068693 + 0.015357752617244813im, 3.0798647688429215e-7 - 5.121387154766871e-6im, 0.026507719691568486),
            atol = 1e-12
        ) |> all

        # check consistency with Julia's Statistics
        @test jackknife(identity, ts)[2] ≈ std(ts)/sqrt(length(ts))
    end
end
