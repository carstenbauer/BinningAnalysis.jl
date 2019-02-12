@testset "Constructors and basic properties" begin
    let F = FullBinner()
        @test typeof(F) <: AbstractVector{Float64}
        @test eltype(F) == Float64
        @test ndims(F) == 1
        @test length(F) == 0
        @test size(F) == (0,)
        @test lastindex(F) == 0
        @test axes(F) == (Base.OneTo(0),)
        @test isempty(F)
    end

    let F = FullBinner(ComplexF64)
        @test typeof(F) <: AbstractVector{ComplexF64}
        @test eltype(F) == ComplexF64
    end

    let x = [0.84, 0.381169, 0.34174, 0.888868, 0.0973183, 0.722725, 0.0957878, 0.432443, 0.755033, 0.864252]
        F = FullBinner(x)
        @test length(F) == 10
        @test lastindex(F) == 10
        @test axes(F) == (Base.OneTo(10),)
        @test !isempty(F)
    end


    let x = [rand(2,3) for _ in 1:100]
        F = FullBinner(x)
        @test length(F) == 100
        @test eltype(F) == Array{Float64,2}
    end
end



@testset "Scalars statistics" begin
    # Real
    let F = FullBinner(1:10_000)
        @test isapprox(std_error(F), 361.4079699881821)
        bs, stds, cum_stds = all_binning_errors(F)
        @test bs == 1:312
        @test isapprox(sum(stds), 106377.96306621947) # take sum as approx. hash
        @test isapprox(sum(cum_stds), 75541.44622415205)
        @test isapprox(tau(F), 77.86159630295694)
    end

    # Test 0/0 bug in R_value
    @test std_error(FullBinner(fill(1.0, 100))) == 0.0

    # Complex
    let F = FullBinner((1:10_000) .+ ((10_000:-1:1) .* im))
        @test isapprox(std_error(F), 511.10805270701564)
        bs, stds, cum_stds = all_binning_errors(F)
        @test bs == 1:312
        @test isapprox(sum(stds), 150441.1581058718) # take sum as approx. hash
        @test isapprox(sum(cum_stds), 106831.73777147368)
        @test isapprox(tau(F), 77.86159630295694)
    end
end




@testset "Arrays statistics" begin
    # Real
    Random.seed!(123)
    let F = FullBinner([rand(2,3) for _ in 1:100])
        @test length(F) == 100
        @test isapprox(std_error(F), [0.029184472105069394 0.029581605926346424 0.027793717502753976; 0.029105387394205307 0.02741415651581391 0.029933054433434834])
        @test isapprox(tau(F), [0.030183772076860294 0.027610002544459222 -0.047279410457739646; -0.003567404598109447 0.03282685243862249 0.06928989962602228])
    end

    # Complex
    Random.seed!(123)
    let F = FullBinner([rand(ComplexF64, 2,3) for _ in 1:100])
        @test length(F) == 100
        @test isapprox(std_error(F), [0.04445492633322362 0.04004496543964919 0.039737207226072296; 0.04099334255252945 0.039004215520294906 0.0409503504806149])
        @test isapprox(tau(F), [0.062160208644724047 -0.005039096051545122 0.037487751473977315; 0.03850424600452085 -0.032173894672710424 0.0320190059417359])
    end
end