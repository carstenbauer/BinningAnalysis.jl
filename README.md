# BinningAnalysis.jl

[![travis][travis-img]](https://travis-ci.org/crstnbr/BinningAnalysis.jl)
[![appveyor][appveyor-img]](https://ci.appveyor.com/project/crstnbr/binninganalysis-jl/branch/master)
[![codecov][codecov-img]](http://codecov.io/github/crstnbr/BinningAnalysis.jl?branch=master)

[travis-img]: https://img.shields.io/travis/crstnbr/BinningAnalysis.jl/master.svg?label=Linux
[appveyor-img]: https://img.shields.io/appveyor/ci/crstnbr/binninganalysis-jl/master.svg?label=Windows
[codecov-img]: https://img.shields.io/codecov/c/github/crstnbr/BinningAnalysis.jl/master.svg?label=codecov

This package implements the following statistical binning tools,

* Logarithmic Binning
* Full Binning
* Jackknife


As per usual, you can install the package with `] add https://github.com/crstnbr/BinningAnalysis.jl`.


### Performance

* Logarithmic Binning
  * Size complexity: $O(\log_2(N))$
  * Time complexity: $O(N)$

where $N$ is the number of values pushed.

---

### Tutorial

```julia
# Create a new logarithmic binner for `Float64`s.
B = LogBinner()
# On default, 2^32-1 values can be added to the binner. This value can be
# changed by passing an integer to LogBinner, e.g. LogBinner(64) for a capacity
# of 2^64-1

# Data can be added with push!,
push!(B, value)
# or append! (multiple values at once)
append!(B, [1,2,3])

# Get the mean and standard error of each binning level
xs  = all_means(B)
Δxs = all_std_errors(B)

# Or get them for an individual binning level
x3  = mean(B, 3)
Δx3 = std_error(B, 3)
# Binning level 0 includes the completely unbinned values. It is not included
# in all_means etc.

# Check whether a level has converged
has_converged(B, 3)
# This checks whether variance/N of level 2 and 3 is approximately the same.
# To be sure that the binning analysis has converged, this criterion should be
# true over multiple levels.
# Note that this criterion is generally not true close to the maximum binning
# level. Usually this is the result of the small effective sample size, rather
# than a convergence failure.

# The autocorrelation time is given by
τ = tau(B, 3)
```
