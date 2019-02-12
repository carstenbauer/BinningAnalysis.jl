module BinningAnalysis

import Statistics: mean, var
import Base: push!, append!, show, summary, eltype,
            isempty, length, ndims, empty!

using Reexport


include("generic.jl")


# LogBinner
include("log/binning.jl")
include("log/cosmetics.jl")
include("log/statistics.jl")
export LogBinner, capacity
export mean, var, varN, tau, std_error
export all_vars, all_varNs, all_taus, all_std_errors
export convergence, has_converged


# "Full" binning
include("full/statistics.jl")
export binning_error
export all_binning_errors


# TSBuffer
include("full/tsbuffer.jl")
export TSBuffer, timeseries, ts



# Jackknife
include("Jackknife.jl")
@reexport using .Jackknife





end # module
