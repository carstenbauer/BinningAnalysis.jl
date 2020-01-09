![logo](https://github.com/crstnbr/BinningAnalysis.jl/blob/master/docs/src/assets/logo_with_text.png)

| **Documentation**                                                               | **Build Status**                                                                                |  **Community**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-dev-img]][docs-dev-url] | ![][lifecycle-img] [![][github-ci-img]][github-ci-url] [![][codecov-img]][codecov-url] | [![][slack-img]][slack-url] [![][license-img]][license-url] [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3603347.svg)](https://doi.org/10.5281/zenodo.3603347) |

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://crstnbr.github.io/BinningAnalysis.jl/dev
[github-ci-img]: https://github.com/crstnbr/BinningAnalysis.jl/workflows/Run%20tests/badge.svg
[github-ci-url]: https://github.com/crstnbr/BinningAnalysis.jl/actions?query=workflow%3A%22Run+tests%22
[codecov-img]: https://img.shields.io/codecov/c/github/crstnbr/BinningAnalysis.jl/master.svg?label=codecov
[codecov-url]: http://codecov.io/github/crstnbr/BinningAnalysis.jl?branch=master

[slack-url]: https://slackinvite.julialang.org/
[slack-img]: https://img.shields.io/badge/chat-on%20slack-yellow.svg
[license-img]: https://img.shields.io/badge/License-MIT-red.svg
[license-url]: https://opensource.org/licenses/MIT

[lifecycle-img]: https://img.shields.io/badge/lifecycle-stable-blue.svg

This package provides tools to estimate [standard errors](https://en.wikipedia.org/wiki/Standard_error) and [autocorrelation times](https://en.wikipedia.org/wiki/Autocorrelation) of correlated time series. A typical example is a [Markov chain](https://en.wikipedia.org/wiki/Markov_chain) obtained in a [Metropolis Monte Carlo simulation](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm).

**Binning tools:**

* Logarithmic Binning
  * Size complexity: `O(log(N))`
  * Time complexity: `O(N)`
* Full Binning (all bin sizes that work out evenly)

**Statistical resampling methods:**

* Jackknife resampling.

<br>

As per usual, you can install the registered package with

```julia
] add BinningAnalysis
```

Note that there is [BinningAnalysisPlots.jl](https://github.com/crstnbr/BinningAnalysisPlots.jl) which defines some [Plots.jl](https://github.com/JuliaPlots/Plots.jl) recipes for `LogBinner` and `FullBinner` to facilitate visualizing the error convergence.

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

If you want to calculate the standard error of an existing time series there you can use the convenience wrapper `std_error(x[; method=:log])`. It takes a keyword argument `method`, which can be `:log`, `:full`, or `:jackknife`.

```julia
ts = rand(1000);
std_error(ts) # default is logarithmic binning
std_error(ts, method=:full)
```


## Supported types

All statistical tools should work with number-like (`<: Number`) and array-like (`<: AbstractArray`) elements. Regarding complex numbers, we follow base Julia and define
`var(x) = var(real(x)) + var(imag(x))`.

If you observe unexpected behavior please file an issue!


## References

* J. Gubernatis, N. Kawashima, and P. Werner, [Quantum Monte Carlo Methods: Algorithms for Lattice Models](https://www.cambridge.org/core/books/quantum-monte-carlo-methods/AEA92390DA497360EEDA153CF1CEC7AC), Book (2016)
* V. Ambegaokar, and M. Troyer, [Estimating errors reliably in Monte Carlo simulations of the Ehrenfest model](http://aapt.scitation.org/doi/10.1119/1.3247985), American Journal of Physics 78, 150 (2010)
