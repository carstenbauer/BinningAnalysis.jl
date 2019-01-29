# BinningAnalysis.jl

[![travis][travis-img]](https://travis-ci.org/crstnbr/BinningAnalysis.jl)
[![appveyor][appveyor-img]](https://ci.appveyor.com/project/crstnbr/binninganalysis-jl/branch/master)
[![codecov][codecov-img]](http://codecov.io/github/crstnbr/BinningAnalysis.jl?branch=master)

[travis-img]: https://img.shields.io/travis/crstnbr/BinningAnalysis.jl/master.svg?label=Linux
[appveyor-img]: https://img.shields.io/appveyor/ci/crstnbr/binninganalysis-jl/master.svg?label=Windows
[codecov-img]: https://img.shields.io/codecov/c/github/crstnbr/BinningAnalysis.jl/master.svg?label=codecov

Install the package with `] add https://github.com/crstnbr/BinningAnalysis.jl.git`.

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
