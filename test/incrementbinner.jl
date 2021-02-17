@testset "Float64" begin
    B = IncrementBinner(blocksize=4)
    
    @test B.keep == 4

    for i in 1:4
        push!(B, 1.0)
    end
    for i in 1:4
        push!(B, 1.0)
        push!(B, 3.0)
    end
    for i in 1:4
        push!(B, 3.0)
        push!(B, 4.0)
        push!(B, 4.0)
        push!(B, 5.0)
    end

    xs = indices(B)
    ys = values(B)

    @test xs == Float64[1,2,3,4, 5.5,7.5,9.5,11.5, 14.5,18.5,22.5,26.5]
    @test ys == Float64[1,1,1,1, 2,2,2,2, 4,4,4,4]
end

@testset "ComplexF64" begin
    B = IncrementBinner(ComplexF64, blocksize=3)
    
    @test B.keep == 3

    for i in 1:3
        push!(B, 1.0+1.0im)
    end
    for i in 1:3
        push!(B, 1.0+3.0im)
        push!(B, 3.0+1.0im)
    end

    xs = indices(B)
    ys = values(B)

    @test xs == Float64[1,2,3, 4.5,6.5,8.5]
    @test ys == ComplexF64[1+1im,1+1im,1+1im, 2+2im,2+2im,2+2im]
end

@testset "Vector" begin
    B = IncrementBinner([0.0, 0.0], blocksize=2)
    
    @test B.keep == 2

    for i in 1:2
        push!(B, [1.0, 1.0])
    end
    for i in 1:2
        push!(B, [1.0, 3.0])
        push!(B, [3.0, 1.0])
    end

    xs = indices(B)
    ys = values(B)

    @test xs == Float64[1,2, 3.5,5.5]
    @test ys == [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]
end