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
    @test BinningAnalysis.unbinned_tau(correlated) ≈ 7.5 rtol = 0.05


    @testset "FullBinner" begin
        B = FullBinner(correlated)

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

            @test mean(B, binsize) ≈ av
            @test var(B, binsize) ≈ _var rtol=0.01
            @test varN(B, binsize) * N_blocks ≈ _var * binsize / N_corr rtol=0.01
            @test std_error(B, binsize) * sqrt(N_corr / binsize) ≈ stderr rtol=0.01
            @test tau(B, binsize) ≈ target rtol=0.01
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

            @test mean(B, binsize) ≈ av
            @test var(B, binsize) ≈ f * _var rtol=0.075
            @test varN(B, binsize) * N_blocks ≈ _var rtol=0.075
            @test std_error(B, binsize) ≈ stderr rtol=0.075
            @test tau(B, binsize) ≈ 0.5(N_corr - 1) rtol = 0.075
        end
    end

    @testset "LogBinner" begin
        B = LogBinner(correlated)

        # Strongly/Maximally correlated
        for lvl in 1:4
            binsize = 2^(lvl-1)
            target = 0.5 * (binsize - 1)

            @test mean(B, lvl) ≈ av
            @test var(B, lvl) ≈ _var rtol=0.01
            @test varN(B, lvl) * N_blocks ≈ _var * binsize / N_corr rtol=0.01
            @test std_error(B, lvl) * sqrt(N_corr / binsize) ≈ stderr rtol=0.01
            @test tau(B, lvl) ≈ target rtol=0.01
        end

        # Not correlated
        for lvl in 5:11
            binsize = 2.0 ^ (lvl-1)
            f = N_corr / binsize

            @test mean(B, lvl) ≈ av
            @test var(B, lvl) ≈ f * _var rtol=0.075
            @test varN(B, lvl) * N_blocks ≈ _var rtol=0.075
            @test std_error(B, lvl) ≈ stderr rtol=0.075
            @test tau(B, lvl) ≈ 0.5(N_corr - 1) rtol = 0.075
        end
    end

    @testset "ErrorPropagator" begin
        B = ErrorPropagator(correlated)

        # Strongly/Maximally correlated
        for lvl in 1:4
            binsize = 2^(lvl-1)
            target = 0.5 * (binsize - 1)

            @test mean(B, 1, lvl) ≈ av
            @test var(B, 1, lvl) ≈ _var rtol=0.01
            @test varN(B, 1, lvl) * N_blocks ≈ _var * binsize / N_corr rtol=0.01
            @test std_error(B, 1, lvl) * sqrt(N_corr / binsize) ≈ stderr rtol=0.01
            @test tau(B, 1, lvl) ≈ target rtol=0.01
        end

        # Not correlated
        for lvl in 5:11
            binsize = 2.0 ^ (lvl-1)
            f = N_corr / binsize

            @test mean(B, 1, lvl) ≈ av
            @test var(B, 1, lvl) ≈ f * _var rtol=0.075
            @test varN(B, 1, lvl) * N_blocks ≈ _var rtol=0.075
            @test std_error(B, 1, lvl) ≈ stderr rtol=0.075
            @test tau(B, 1, lvl) ≈ 0.5(N_corr - 1) rtol = 0.075
        end
    end
end