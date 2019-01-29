module BinningAnalysis

import Statistics: mean, var
import Base: push!, append!, show, summary, eltype,
            isempty, length, ndims, empty!

include("binning.jl")
export LogBinner

include("statistics.jl")
export mean, var, varN, tau, std_error
export all_vars, all_varNs, all_taus, all_std_errors
export convergence, has_converged

include("cosmetics.jl")

end # module
