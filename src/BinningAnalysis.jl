module BinningAnalysis

import Statistics: mean, var
import Base.push!


include("binning.jl")
export BinnerA, push!

include("statistics.jl")
export mean, var, varN, tau, std_error
export all_means, all_vars, all_varNs, all_taus, all_std_errors
export convergence, has_converged

end # module
