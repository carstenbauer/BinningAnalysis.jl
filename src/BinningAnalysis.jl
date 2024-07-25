module BinningAnalysis

import Statistics: mean, var
using Reexport, Lazy, RecursiveArrayTools


# Generic functions
include("generic.jl")
export std_error, tau


# Logarithmic binning (keeping log(N) data points)
include("log/accumulators.jl")
include("log/binning.jl")
include("log/statistics.jl")
export LogBinner, capacity
export mean, var, varN, tau, std_error, autocorrelation, autocorrelation_time
export all_vars, all_varNs, all_taus, all_std_errors, all_autocorrelations, all_autocorrelation_times
export convergence, has_converged


# Full binning (keeping all data points)
include("full/binning.jl")
include("full/statistics.jl")
export FullBinner


# Jackknife resampling (estimate function result variance via resampling)
include("Jackknife.jl")
@reexport using .Jackknife


# log binning with covariance matrix to estimate variance of function result
include("ErrorPropagation/binning.jl")
include("ErrorPropagation/statistics.jl")
export ErrorPropagator
export means, vars, varNs, taus, std_errors, covmat, all_means


# temporal binning with increasing bin size for estimating convergence
include("incrementing/IncrementBinner.jl")
export IncrementBinner
export indices


# (expensive) autocorrelation estimation via progression of correlation
include("direct_tau.jl")


# Wrapper for Binners which accumulates N-element means before passing to binners 
# (compression of low binning levels)
include("PreBinner.jl")
export PreBinner

end # module
