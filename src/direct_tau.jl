function correlation(sample, k)
    # Stabilized correlation function χ(k) = χ(|i-j|) = ⟨xᵢxⱼ⟩ - ⟨xᵢ⟩⟨xⱼ⟩
    # Following Welford derivation from https://changyaochen.github.io/welford/

    M = length(sample)
    corr = sumx = sumy = 0.0
    for i in 1:M-k
        # ̅xᵢ₊₁ = ̅xᵢ + (xᵢ₊₁ - ̅xᵢ) / (N+1)
        # ̅yᵢ₊₁ = ̅yᵢ + (yᵢ₊₁ - ̅yᵢ) / (N+1)
        # χᵢ₊₁ = χᵢ + [(xᵢ₊₁ - ̅xᵢ)(yᵢ₊₁ - ̅yᵢ₊₁) - χᵢ] / (i + 1)

        invN = 1.0 / i
        sumy += invN * (sample[i + k] - sumy)
        sumx_delta = invN * (sample[i] - sumx)
        corr += sumx_delta * (sample[i + k] - sumy) - invN * corr
        sumx += sumx_delta
    end
    return corr
end

# QMCM eq 3.15
"""
    unbinned_tau(sample[; truncate = true, max_rel_err = 0.0, min_sample_size = 32])

Estimates the autocorrelation time `τ = ∑ₖ (1 - k/N) χₖ/χ₀` from a sample using the 
correlation function `χₖ = ⟨xᵢ xᵢ₊ₖ⟩ - ⟨xᵢ⟩⟨xᵢ₊ₖ⟩` where x are values in the sample.

Note that by default this function truncated the sum over k if 
`v / τ < max_rel_err / (N - k)` or if the `k > N - min_sample_size`. The former 
cuts of fluctuations around 0 and can be turned of with `truncate = false`.
"""
function unbinned_tau(sample; truncate = true, max_rel_err = 0.0, min_sample_size = 32)
    tau = 0.0

    for k in 1:length(sample) - max(1, min_sample_size)
        v = (1 - k/length(sample)) * correlation(sample, k)
        tau += v

        # We assume the worst case is a constant correlation for the tail.
        # There are (length(sample) - k) values left
        # The average prefactor is 0.5 * (1 - k/length(sample))
        # If the sum of the tail is smaller than max_rel_err * tau, i.e. if tau
        # can at most increase by a factor of max_rel_error, we stop the loop
        if truncate && 0.5 * v * (length(sample) - k) < max_rel_err * tau
            @debug "Cancelled unbinned_tau summation after $k iterations."
            break
        end
    end

    return tau / correlation(sample, 0)
end