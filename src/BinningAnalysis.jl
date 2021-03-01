module BinningAnalysis

import Statistics: mean, var
using Reexport, Lazy, RecursiveArrayTools


# Generic functions
include("generic.jl")
export std_error, tau


# Logarithmic binning
include("log/accumulators.jl")
include("log/binning.jl")
include("log/statistics.jl")
export LogBinner, capacity
export mean, var, varN, tau, std_error
export all_vars, all_varNs, all_taus, all_std_errors
export convergence, has_converged


# "Full" binning
include("full/binning.jl")
include("full/statistics.jl")
export FullBinner, std_error, all_binning_errors


# Jackknife resampling
include("Jackknife.jl")
@reexport using .Jackknife


include("ErrorPropagation/binning.jl")
include("ErrorPropagation/statistics.jl")
export ErrorPropagator
export mean, var, varN, tau, std_error
export means, vars, varNs, taus, std_errors
export all_means, all_vars, all_varNs, all_taus, all_std_errors
export covmat


include("incrementing/IncrementBinner.jl")
export IncrementBinner
export indices

end # module
