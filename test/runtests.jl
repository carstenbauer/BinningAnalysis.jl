using BinningAnalysis
using Test, Statistics, Random


@testset "All Tests" begin

    @testset "Logarithmic Binning" begin
        include("logbinning.jl")
    end


    @testset "Full Binning" begin
        include("fullbinning.jl")
    end


    @testset "Generic functions" begin
        x = [0.0561823, 0.846613, 0.813439, 0.357134, 0.157445, 0.103298, 0.948842, 0.629425, 0.290206, 0.00695332, 0.869828, 0.949165, 0.897995, 0.916239, 0.457564, 0.349827, 0.398683, 0.264218, 0.72754, 0.934315, 0.666448, 0.134813, 0.364933, 0.829088, 0.256443, 0.595029, 0.172097, 0.241686, 0.489935, 0.239663, 0.391291, 0.00751015, 0.138935, 0.0569876, 0.571786, 0.694996, 0.798602, 0.923308, 0.73978, 0.414774, 0.835145, 0.731303, 0.271647, 0.707796, 0.00348624, 0.0905812, 0.316176, 0.921054, 0.131037, 0.599667, 0.805071, 0.440813, 0.086516, 0.363658, 0.476161, 0.931257, 0.28974, 0.78717, 0.60822, 0.144024, 0.214432, 0.061922, 0.626495, 0.512072, 0.758078, 0.840485, 0.242576, 0.147441, 0.599222, 0.993569, 0.0365044, 0.0983033, 0.713144, 0.422394, 0.480044, 0.968745, 0.518475, 0.431319, 0.4432, 0.526007, 0.612975, 0.468387, 0.262145, 0.888011, 0.105744, 0.325821, 0.769525, 0.289073, 0.336083, 0.443037, 0.489698, 0.141654, 0.915284, 0.319068, 0.341001, 0.704346, 0.0794996, 0.0412352, 0.70016, 0.0195158]
        @test isapprox(std_error(x), std_error(x; method=:log))
        @test isapprox(std_error(x; method=:log), 0.03384198827323159)
        @test isapprox(std_error(x; method=:full), 0.030761063167290947)
        @test isapprox(std_error(x; method=:jackknife), 0.029551526503551463)

        x = x .+ 1im
        @test isapprox(std_error(x), std_error(x; method=:log))
        @test isapprox(std_error(x; method=:log), 0.0338419882732316)
        @test isapprox(std_error(x; method=:full), 0.03076106316729094)
        @test isapprox(std_error(x; method=:jackknife), 0.029551526503551445)


        x = Array{Float64,2}[[0.311954 0.476706 0.314101; 0.420978 0.478085 0.0194284], [0.491386 0.493583 0.295477; 0.136896 0.1634 0.84641], [0.318536 0.705903 0.121377; 0.764174 0.240484 0.25894], [0.286923 0.172458 0.392881; 0.124348 0.140628 0.730131], [0.287451 0.221914 0.382938; 0.29568 0.249575 0.87685]]
        @test isapprox(std_error(x), std_error(x; method=:log))
        @test isapprox(std_error(x, method=:log), [0.03856212859659066 0.09765048982846934 0.04879645809318544; 0.11744109653881814 0.05978167652935139 0.1722342286233907])
        @test isapprox(std_error(x, method=:full), [0.03856212859659072 0.09765048982846936 0.048796458093185405; 0.11744109653881814 0.05978167652935137 0.17223422862339066])

        x = ["this", "should", "error"]
        @test_throws ErrorException std_error(x, method=:jackknife)
        @test_throws ArgumentError std_error(x, method=:whatever)
    end



    @testset "Jackknife" begin
        g(x1, x2) = x1^2 - x2

        # Real
        ts = [0.00124803, 0.643089, 0.183268, 0.799899, 0.0857666, 0.955348, 0.165763, 0.765998, 0.63942, 0.308818]
        ts2 = [0.606857, 0.0227746, 0.805997, 0.978731, 0.0853112, 0.311463, 0.628918, 0.0190664, 0.515998, 0.0223728]
        @test isapprox.(
            jackknife_full(identity, ts),
            (0.45486176300000025, 0.0, 0.10834619757501414),
            atol = 1e-12
        ) |> all
        @test isapprox.(
            jackknife_full(identity, 1 ./ ts),
            (83.4370988407286, 0.0, 79.76537738034833),
            atol = 1e-12
        ) |> all
        @test isapprox.(
            jackknife_full(g, ts, ts2.^2),
            (-0.07916794438503649, 0.011738898528964592, 0.14501699232741938),
            atol = 1e-12
        ) |> all
        @test jackknife_full(g, hcat(ts, ts2.^2)) == jackknife_full(g, ts, ts2.^2)

        # check consistency with Julia's Statistics
        @test jackknife(identity, ts)[2] ≈ std(ts)/sqrt(length(ts))

        # Complex
        ts = Complex{Float64}[0.0259924+0.674798im, 0.329853+0.558688im, 0.821612+0.142805im, 0.0501703+0.801068im, 0.0309707+0.877745im, 0.937856+0.852463im, 0.669084+0.606286im, 0.887004+0.431615im, 0.763452+0.210563im, 0.678384+0.428294im]
        ts2 = Complex{Float64}[0.138147+0.11007im, 0.484956+0.127761im, 0.986078+0.702827im, 0.45161+0.878256im, 0.768398+0.954537im, 0.228518+0.260732im, 0.256892+0.437918im, 0.647672+0.749172im, 0.0587658+0.408715im, 0.0792651+0.288578im]
        @test isapprox.(
            jackknife_full(identity, ts),
            (
                0.519437840000001 + 0.5584324999999994im,
                -9.992007221626409e-16 + 0.0im,
                0.1428607353199917
            ),
            atol = 1e-12
        ) |> all
        @test isapprox.(
            jackknife_full(identity, 1 ./ ts),
            (
                0.6727428408881817 - 0.8112743686359876im,
                -9.992007221626409e-16 + 1.9984014443252818e-15im,
                0.20473472232587642
            ),
            atol = 1e-12
        ) |> all
        @test isapprox.(
            jackknife_full(g, ts, ts2.^2),
            (
                0.021446672721215643 + 0.06979705636190503im,
                0.007300401672734998 - 0.01055311354806504im,
                0.2752976586383889
            ),
            atol = 1e-12
        ) |> all
        @test jackknife_full(g, hcat(ts, ts2.^2)) == jackknife_full(g, ts, ts2.^2)

        # check consistency with Julia's Statistics
        @test jackknife(identity, ts)[2] ≈ std(ts)/sqrt(length(ts))
    end
end
