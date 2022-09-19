@testset "Constructors and basic properties" begin
    let 
        F = FullBinner()
        F2 = FullBinner()
        @test typeof(F) <: BinningAnalysis.AbstractBinner{Float64}
        @test eltype(F) == Float64
        @test ndims(F) == 0
        @test length(F) == 0
        @test_throws BoundsError size(F)
        @test lastindex(F) == 0
        @test axes(F) == (Base.OneTo(0),)
        @test isempty(F)
        @test F == F2
        @test F ≈ F2
    end

    let F = FullBinner(ComplexF64)
        @test typeof(F) <: BinningAnalysis.AbstractBinner{ComplexF64}
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
    # Real, Complex
    for F in (
            FullBinner(collect(1:10_000)),
            FullBinner(collect((1:10_000) .+ ((10_000:-1:1) .* im)))
        )

        # Unbinned Statistics vs Base
        µ = mean(F.x)
        v = var(F.x, mean = µ)
        Δµ = sqrt(v / 10_000)

        @test mean(F, 1) ≈ µ
        @test var(F, 1) ≈ v
        @test varN(F, 1) ≈ v / 10_000
        @test std_error(F, 1) ≈ Δµ

        # Binned stats (these values are strongly correlated)
        binsize = BinningAnalysis._reliable_level(F)
        @test binsize == div(length(F), 32)
        @test BinningAnalysis._eachlevel(F) == 1:binsize
        @test mean(F) ≈ µ
        @test var(F) ≈ v rtol = 0.05
        @test std_error(F) ≈ Δµ * sqrt(binsize) rtol = 0.05
        @test tau(F) ≈ 0.5(binsize-1) rtol = 0.05
    end
end




@testset "Arrays statistics" begin
    for T in (Float64, ComplexF64)
        StableRNGs.seed!(rng, 123)
        F = FullBinner([rand(rng, T, 2, 3) for _ in 1:1000])

        # Unbinned Statistics vs Base
        µ = mean(F.x)
        v = var(F.x, mean = µ)
        Δµ = sqrt.(v ./ 1000)

        @test mean(F, 1) ≈ µ
        @test var(F, 1) ≈ v
        @test varN(F, 1) ≈ v ./ 1000
        @test std_error(F, 1) ≈ Δµ

        # Binned stats (these values are uncorrelated)
        # Note that the errors can be quite large due to the relatively small 
        # sample size
        binsize = BinningAnalysis._reliable_level(F)
        @test binsize == div(length(F), 32)
        @test BinningAnalysis._eachlevel(F) == 1:binsize
        @test mean(F) ≈ µ
        @test var(F) ≈ v/binsize rtol = 0.25
        @test std_error(F) ≈ Δµ rtol = 0.15
        @test all(tau(F) .< 1)

        if T == Float64
            @test var(F) ≈ [0.003208002497613544 0.0029222442318581378 0.002097985378538715; 0.0027042522201602293 0.0035246447808234393 0.003948312674521377]
        else
            @test var(F) ≈ [0.006091716418175418 0.0052399001679973665 0.007313376580285079; 0.0059813002774426445 0.006343840253127798 0.004989114094997521]
        end
    end

    F = FullBinner(Vector{Float64})
    x = Float64[1, 2, 3]
    push!(F, x)
    x .= 5
    push!(F, x)
    @test F.x[1] == Float64[1, 2, 3]
    @test F.x[2] == Float64[5, 5, 5]
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