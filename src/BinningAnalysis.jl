module BinningAnalysis

@static if VERSION < v"0.7"
    import Base: mean, var
else
    import Statistics: mean, var
end
import Base.push!


include("binning.jl")
export BinnerA, push!

include("statistics.jl")
export mean, var, varN, tau, std_error
export all_vars, all_varNs, all_taus, all_std_errors
export convergence, has_converged

end # module
