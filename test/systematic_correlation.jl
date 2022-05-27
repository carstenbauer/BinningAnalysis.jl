@testset "Checking converging data (Real)" begin
    # block of maximally correlated values:
    N_corr = 16

    # number of blocks
    N_blocks = 131_072 # 2^17

    uncorrelated = rand(N_blocks) # rand(rng, N_blocks)
    av = mean(uncorrelated)
    _var = var(uncorrelated)
    stderr = sqrt(_var/N_blocks)

    correlated = [x for x in uncorrelated for _ in 1:N_corr]

    @testset "FullBinner" begin
        BA = FullBinner(correlated)

        # Strongly/Maximally correlated

        # Note:
        # With xs = [sample, sample] we have
        # 1/N sum(xs) = 1/N (sum(xs[1:N/2] + sum(xs[N/2+1:N])) = 2/N sum(xs[1:N/2])
        # thus both the mean and variance should not vary when repeating data
        # this doesn't seem to be exactly true for Statistics.var(vcat(sample, sample))
        # this isn't exactly true for BinningAnalysis due to truncation at odd binning levels
        
        # Expectations:
        # - mean matches
        # - var matches up to binsize = N_corr for power binsizes
        # - varN = var / N
        # - std_error = sqrt(var / N) grows with sqrt(N)
        # - autocorrelation time should be 0.5 * (binsize-1) (i.e. maximal)

        for binsize in 2 .^ (0:3)
            target = 0.5binsize - 0.5

            @test mean(BA, binsize) ≈ av
            @test var(BA, binsize) ≈ _var rtol=0.01
            @test varN(BA, binsize) * N_blocks ≈ _var * binsize / N_corr rtol=0.01
            @test std_error(BA, binsize) * sqrt(N_corr / binsize) ≈ stderr rtol=0.01
            @test tau(BA, binsize) ≈ target rtol=0.01
        end

        # Not correlated

        # Note:
        # If we expand the variance of pair binned data
        # \frac{1}{N} \sum_{i=1}^N (\frac{x_{2i-1} + x_{2i}}{2} - \mu)^2
        # and pull out jalf the unbinned variance
        # \frac{1}{2} \frac{1}{2N} \sum_{i=1}^{2N} (x_i - \mu)^2
        # we are left with an additional term
        # \frac{1}{2} \frac{1}{N} \sum_{i=1}^N (x_{2i-1} x_{2i} - 2\mu (x_{2i-1} x_{2i}) + \mu^2)
        # if we assume no correlation we can simplify the first term as
        # \langle x_{2i-1} x_{2i} \rangle = \langle x_{2i-1} \rangle \langle x_{2i} \rangle = \mu^2
        # and the term drops, leaving us with
        # var(x_{pair binned}) ≈ \frac{1}{2} var(x)

        # At and after binning out correlations
        # - mean matches
        # - var ≈ unbinned var / 2
        # - varN = var / N with N = N_blocks * N_corr / binsize
        # - std_error = sqrt(var / N) should be constant as the change in N is 
        #                             cancelled by the change in var
        # - tau should be roughly constant

        for binsize in 2 .^ (4:10)
            f = N_corr / binsize

            @test mean(BA, binsize) ≈ av
            @test var(BA, binsize) ≈ f * _var rtol=0.075
            @test varN(BA, binsize) * N_blocks ≈ _var rtol=0.075
            @test std_error(BA, binsize) ≈ stderr rtol=0.075
            @test tau(BA, binsize) ≈ 0.5(N_corr - 1) rtol = 0.075
        end
    end

    @testset "LogBinner" begin
        BA = LogBinner(correlated)

        # Strongly/Maximally correlated
        for lvl in 1:4
            binsize = 2^(lvl-1)
            target = 0.5 * (binsize - 1)

            @test mean(BA, lvl) ≈ av
            @test var(BA, lvl) ≈ _var rtol=0.01
            @test varN(BA, lvl) * N_blocks ≈ _var * binsize / N_corr rtol=0.01
            @test std_error(BA, lvl) * sqrt(N_corr / binsize) ≈ stderr rtol=0.01
            @test tau(BA, lvl) ≈ target rtol=0.01
        end

        # Not correlated
        for lvl in 5:11
            binsize = 2.0 ^ (lvl-1)
            f = N_corr / binsize

            @test mean(BA, lvl) ≈ av
            @test var(BA, lvl) ≈ f * _var rtol=0.05
            @test varN(BA, lvl) * N_blocks ≈ _var rtol=0.05
            @test std_error(BA, lvl) ≈ stderr rtol=0.05
            @test tau(BA, lvl) ≈ 0.5(N_corr - 1) rtol = 0.05
        end
    end
end