[![travis][travis-img]](https://travis-ci.org/crstnbr/BinningAnalysis.jl)
[![appveyor][appveyor-img]](https://ci.appveyor.com/project/crstnbr/binninganalysis-jl/branch/master)
[![codecov][codecov-img]](http://codecov.io/github/crstnbr/BinningAnalysis.jl?branch=master)
[![license: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![coveralls][coveralls-img]](https://coveralls.io/github/crstnbr/BinningAnalysis.jl?branch=master) !-->

[travis-img]: https://img.shields.io/travis/crstnbr/BinningAnalysis.jl/master.svg?label=linux
[appveyor-img]: https://img.shields.io/appveyor/ci/crstnbr/binninganalysis-jl/master.svg?label=windows
[codecov-img]: https://img.shields.io/codecov/c/github/crstnbr/BinningAnalysis.jl/master.svg?label=codecov
[coveralls-img]: https://img.shields.io/coveralls/github/crstnbr/BinningAnalysis.jl/master.svg?label=coverage

![logo](https://github.com/crstnbr/BinningAnalysis.jl/blob/master/docs/src/assets/logo_with_text.png)

This package implements the following binning tools,

* Logarithmic Binning (most performant)
  * Size complexity: $\mathcal{O}(\log_2(N))$
  * Time complexity: $\mathcal{O}(N)$
* Full Binning

and the following statistical resampling methods,

* Jackknife resampling.

As per usual, you can install the package with

```julia
] add https://github.com/crstnbr/BinningAnalysis.jl
```

## Binning tools

### Logarithmic Binning

```julia
B = LogBinner()
# As per default, 2^32-1 ≈ 4 billion values can be added to the binner. This value can be
# tuned with the `capacity` keyword argument.

push!(B, 4.2)
append!(B, [1,2,3]) # multiple values at once

x  = mean(B)
Δx = std_error(B) # standard error of the mean
tau_x = tau(B) # autocorrelation time

# Alternatively you can provide a time series already in the constructor
x = rand(100)
B = LogBinner(x)

Δx = std_error(B)
```

<!--
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
!-->

### Full Binning

```julia
B = FullBinner() # <: AbstractVector (lightweight wrapper)

push!(B, 2.0)
append!(B, [1,2,3])

x  = mean(B)
Δx = std_error(B) # standard error of the mean

# Alternatively you can provide a time series already in the constructor
x = rand(100)
F = FullBinner(x)

push!(F, 2.0) # will modify x as F is just a thin wrapper

Δx = std_error(F)
```

## Resampling methods

### Jackknife

```julia
x = rand(100)

xmean, Δx = jackknife(identity, x) # jackknife estimates for mean and standard error of <x>

# in this example
# isapprox(Δx, std(x)/sqrt(length(x))) == true

x_inv_mean, Δx_inv = jackknife(identity, 1 ./ x) # # jackknife estimates for mean and standard error of <1/x>

# Multiple time series
x = rand(100)
y = rand(100)

# The inputs of the function `g` must be provided as arguments in `jackknife`.
g(x, y, xy) = x * y / xy  # <x><y> / <xy>
g_mean, Δg = jackknife(g, x, y, x .* y)
```


## Convenience wrapper

If you have a time series and you simply want to calculate the standard error, using any of the aboves methods, there is a convenience wrapper `std_error(x[; method=:log])`. It takes a keyword argument `method` which can be `:log`, `:full`, or `:jackknife`.

```julia
ts = rand(1000);
std_error(ts) # default is logarithmic binning
std_error(ts, method=:full)
```


## Supported types

All statistical tools should work with number-like (`<: Number`) and array-like (`<: AbstractArray`) elements. Regarding complex numbers, we follow base Julia and define
`var(x) = var(real(x)) + var(imag(x))`.

If you observe unexpected behavior please file an issue!
