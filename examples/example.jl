using PyPlot, BinningAnalysis

function make_plot(xs, label)
    figure()
    title(label)
    plot(eachindex(xs), xs)
    xlabel("Binnging level")
    ylabel(label)
end


# Create Binning Analysis
BA = LogBinner()

# Get some data
N_corr = 16
N_blocks = 131_072
for _ in 1:N_blocks
    x = rand()
    for __ in 1:N_corr
        push!(BA, x)
    end
end

means = all_means(BA)
vars = all_vars(BA)
varNs = all_varNs(BA)
errors = all_std_errors(BA)
taus = all_taus(BA)

# Means should be constant across all levels
make_plot(means, "mean")

# The variance decays with increasing lvl, because the sample size decreases.
make_plot(vars, "Var")

# varN is the Variance normalized to the sample size. It increases for correlated
# measurements and stays roughly constant for uncorrelated ones. It fluctuates
# more with decreasing sample size (i.e. increasing binning level)
make_plot(varNs, "Var/N")

# The standard error gives the error of the mean. It behaves like varN.
make_plot(errors, "Standard Error")

# The auto correlation time should be 16... ?
make_plot(taus, "Autocorrelation Time")


# Convergence:
# Green dots are accepted as converged, red dots are not. In general convergence
# should be accepted when the Var/N becomes roughly constant
make_plot(varNs, "Var/N")
for lvl in 1:20
    tag = if has_converged(BA, lvl)
        "go"
    else
        "ro"
    end
    plot([lvl-1], [varNs[lvl]], tag)
end
