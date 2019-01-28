# BinningAnalysis

[![Build Status](https://travis-ci.org/ffreyer/BinningAnalysis.jl.svg?branch=master)](https://travis-ci.org/ffreyer/BinningAnalysis.jl)
[![Coverage Status](https://coveralls.io/repos/ffreyer/BinningAnalysis.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/ffreyer/BinningAnalysis.jl?branch=master)
[![codecov.io](http://codecov.io/github/ffreyer/BinningAnalysis.jl/coverage.svg?branch=master)](http://codecov.io/github/ffreyer/BinningAnalysis.jl?branch=master)

Install the package with `Pkg.add("https://github.com/ffreyer/BinningAnalysis.jl.git")`

---

#### Performance

* Size complexity: `O(log(N))`
* Time complexity: `O(N)`

where N is the number of values pushed.

---

#### Tutorial

```julia
# Create a new binning analysis.
binner = LogBinner()
# On default, 2^32-1 values can be added to the binner. This value can be
# changed by passing an integer to LogBinner, e.g. LogBinner(64) for a capacity
# of 2^64-1

for value in data
    # Data can be added with push!.
    push!(binner, value)
    # An error will be thrown if push exceeds the capactiy of the binner.
end

# Get the mean and standard error of each binning level
xs  = all_means(binner)
Δxs = all_std_errors(binner)

# Or get them for an individual binning level
x3  = mean(binner, 3)
Δx3 = std_error(binner, 3)
# Binning level 0 includes the completely unbinned values. It is not included
# in all_means etc.

# Check whether a level has converged
has_converged(binner, 3)
# This checks whether variance/N of level 2 and 3 is approximately the same.
# To be sure that the binning analysis has converged, this criterion should be
# true over multiple levels.
# Note that this criterion is generally not true close to the maximum binning
# level. Usually this is the result of the small effective sample size, rather
# than a convergence failure.

# The autocorrelation time is given by
τ = tau(binner, 3)
```
