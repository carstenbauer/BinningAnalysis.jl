# BinningAnalysis.jl

[![travis][travis-img]](https://travis-ci.org/crstnbr/BinningAnalysis.jl)
[![appveyor][appveyor-img]](https://ci.appveyor.com/project/crstnbr/binninganalysis-jl/branch/master)
[![codecov][codecov-img]](http://codecov.io/github/crstnbr/BinningAnalysis.jl?branch=master)

[travis-img]: https://img.shields.io/travis/crstnbr/BinningAnalysis.jl/master.svg?label=Linux
[appveyor-img]: https://img.shields.io/appveyor/ci/crstnbr/binninganalysis-jl/master.svg?label=Windows
[codecov-img]: https://img.shields.io/codecov/c/github/crstnbr/BinningAnalysis.jl/master.svg?label=codecov

This package implements the following statistical binning tools,

* Logarithmic Binning
  * Size complexity: O(log<sub>2</sub>(N))
  * Time complexity: O(N)
* Jackknife
<!-- * Full Binning -->

As per usual, you can install the package with `] add https://github.com/crstnbr/BinningAnalysis.jl`.

---

### Tutorial: Logarithmic Binning

```julia
# Create a new logarithmic binner for `Float64`s.
B = LogBinner()
# On default, 2^32-1 ≈ 4 billion values can be added to the binner. This value can be
# tuned with the `capacity` keyword argument.

# Data can be added with push!,
push!(B, value)
# or append! (multiple values at once)
append!(B, [1,2,3])

# Get the mean, standard error, and autocorrelation estimates
x  = mean(B)
Δx = std_error(B)
tau_x = tau(B)

# You can also get the standard error estimates for all binning levels individually.
Δxs = all_std_errors(B)

# BETA: Check whether a level has converged
has_converged(B, 3)
# This checks whether variance/N of level 2 and 3 is approximately the same.
# To be sure that the binning analysis has converged, this criterion should be
# true over multiple levels.
# Note that this criterion is generally not true close to the maximum binning
# level. Usually this is the result of the small effective sample size, rather
# than a convergence failure.
```

### Tutorial: Jackknife

```julia
x = rand(100) # a time series

# Let's start with a trivial example, the jackknife standard error of mean(x)
Δx = jackknife(mean, x)

# This is, of course, equal to std(mean(x))/sqrt(length(x)) up to numerical precision
isapprox(jackknife(mean, ts), std(ts)/sqrt(length(ts))) == true

# However, we can use any function of the time series in `jackknife`. For example,
# we can calculate the jackknife standard error of the inverse.
Δx_inv = jackknife(x -> mean(1 ./ x), x)

# We can also calculate standard error estimates of observables calculated from many time series
x = rand(100)
y = rand(100)

# The input z will be a matrix whose columns correspond to x and y, i.e. z[:,1] == x and z[:,2] == y
g(z) = @views mean(z[:,1]) * mean(z[:,2]) / mean(z[:,1] .* z[:,2])  # <x><y> / <xy>
Δg = jackknife(g, x, y)
```
