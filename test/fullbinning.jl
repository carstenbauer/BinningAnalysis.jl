@testset "Constructors and basic properties" begin
    let 
        F = FullBinner()
        F2 = FullBinner()
        @test typeof(F) <: AbstractVector{Float64}
        @test eltype(F) == Float64
        @test ndims(F) == 1
        @test length(F) == 0
        @test size(F) == (0,)
        @test lastindex(F) == 0
        @test axes(F) == (Base.OneTo(0),)
        @test isempty(F)
        @test F == F2
        @test F ≈ F2
    end

    let F = FullBinner(ComplexF64)
        @test typeof(F) <: AbstractVector{ComplexF64}
        @test eltype(F) == ComplexF64
    end

    let x = [0.84, 0.381169, 0.34174, 0.888868, 0.0973183, 0.722725, 0.0957878, 0.432443, 0.755033, 0.864252]
        F = FullBinner(x)
        F2 = FullBinner(Float64)
        @test length(F) == 10
        @test lastindex(F) == 10
        @test axes(F) == (Base.OneTo(10),)
        @test !isempty(F)
        @test F != F2
        @test !(F ≈ F2)
        append!(F2, x)
        @test F == F2
        @test F ≈ F2
    end


    let x = [rand(rng, 2,3) for _ in 1:100]
        F = FullBinner(x)
        @test length(F) == 100
        @test eltype(F) == Array{Float64,2}
    end
end



@testset "Scalars statistics" begin
    # Real
    StableRNGs.seed!(rng, 123)
    let F = FullBinner(1:10_000)
        @test isapprox(std_error(F), 361.4079699881821)
        bs, stds, cum_stds = all_binning_errors(F)
        @test bs == 1:312
        @test isapprox(sum(stds), 106377.96306621947) # take sum as approx. hash
        @test isapprox(sum(cum_stds), 75541.44622415205)
        @test isapprox(tau(F), 77.86159630295694)

        # beta: convergence
        @test !BinningAnalysis.isconverged(F)
    end

    # Test 0/0 bug in R_value
    @test std_error(FullBinner(fill(1.0, 100))) == 0.0

    # Complex
    StableRNGs.seed!(rng, 123)
    let F = FullBinner((1:10_000) .+ ((10_000:-1:1) .* im))
        @test isapprox(std_error(F), 511.10805270701564)
        bs, stds, cum_stds = all_binning_errors(F)
        @test bs == 1:312
        @test isapprox(sum(stds), 150441.1581058718) # take sum as approx. hash
        @test isapprox(sum(cum_stds), 106831.73777147368)
        @test isapprox(tau(F), 77.86159630295694)

        # beta: convergence
        @test !BinningAnalysis.isconverged(F)
    end

    # R -> tau conversion
    @test BinningAnalysis._tau(2.4) == 0.7
end




@testset "Arrays statistics" begin
    # Real
    StableRNGs.seed!(rng, 123)
    let F = FullBinner([rand(rng, 2,3) for _ in 1:100])
        @test length(F) == 100
        @test isapprox(std_error(F), [0.03143740602417682 0.027704526977409927 0.030854857036671797; 0.030644671905884845 0.027310786213237917 0.027354755729282157])
        @test isapprox(tau(F), [0.10870021011145892 0.0061711648482875026 0.11027212856462598; 0.01378233853009625 -0.042306404021438704 -0.007019406166005104])
    end

    # Complex
    StableRNGs.seed!(rng, 123)
    let F = FullBinner([rand(rng, ComplexF64, 2,3) for _ in 1:100])
        @test length(F) == 100
        @test isapprox(std_error(F), [0.04307722433951857 0.038921014462829306 0.04216835617026453; 0.04084767913236551 0.04298458677617402 0.03860867715515041])
        @test isapprox(tau(F), [0.043561275772290076 -0.024223023364989493 -0.019126929690185812; 0.006471156025580127 0.05209550936082041 -0.06035050880950349])
    end
end





@testset "Cosmetics (show, print, etc.)" begin
    F = FullBinner();
    # empty binner
    oldstdout = stdout
    (read_pipe, write_pipe) = redirect_stdout()
    println(F) # compact
    show(write_pipe, MIME"text/plain"(), F) # full
    redirect_stdout(oldstdout);
    close(write_pipe);

    # compact
    if VERSION < v"1.7.0"
        @test readline(read_pipe) == "FullBinner{Float64,Array{Float64,1}}()"
        @test readline(read_pipe) == "FullBinner{Float64,Array{Float64,1}}"
    else
        @test readline(read_pipe) == "FullBinner{Float64,Vector{Float64}}()"
        @test readline(read_pipe) == "FullBinner{Float64,Vector{Float64}}"
    end
    # full
    @test readline(read_pipe) == "| Count: 0"
    @test length(readlines(read_pipe)) == 0
    close(read_pipe);

    # filled binner
    StableRNGs.seed!(rng, 123)
    append!(F, rand(rng, 1000))
    (read_pipe, write_pipe) = redirect_stdout()
    show(write_pipe, MIME"text/plain"(), F)
    redirect_stdout(oldstdout);
    close(write_pipe);
    @test if VERSION < v"1.7.0"
        readline(read_pipe) == "FullBinner{Float64,Array{Float64,1}}"
    else
        readline(read_pipe) == "FullBinner{Float64,Vector{Float64}}"
    end
    @test readline(read_pipe) == "| Count: 1000"
    @test readline(read_pipe) == "| Mean: 0.49387"
    @test length(readlines(read_pipe)) == 0
    close(read_pipe);
end