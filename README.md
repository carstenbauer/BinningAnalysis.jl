![logo](https://github.com/carstenbauer/BinningAnalysis.jl/blob/master/docs/src/assets/logo_with_text.png)

| **Build Status**                                                                                |  **Community**                                                                                |
|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| ![][lifecycle-img] [![][github-ci-img]][github-ci-url] [![][codecov-img]][codecov-url] [![][pkgeval-img]][pkgeval-url] | [![][slack-img]][slack-url] [![][license-img]][license-url] [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3603347.svg)](https://doi.org/10.5281/zenodo.3603347) |

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://carstenbauer.github.io/BinningAnalysis.jl/dev
[github-ci-img]: https://github.com/carstenbauer/BinningAnalysis.jl/workflows/Run%20tests/badge.svg
[github-ci-url]: https://github.com/carstenbauer/BinningAnalysis.jl/actions?query=workflow%3A%22Run+tests%22
[codecov-img]: https://img.shields.io/codecov/c/github/carstenbauer/BinningAnalysis.jl/master.svg?label=codecov
[codecov-url]: http://codecov.io/github/carstenbauer/BinningAnalysis.jl?branch=master

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/B/BinningAnalysis.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/report.html

[slack-url]: https://slackinvite.julialang.org/
[slack-img]: https://img.shields.io/badge/chat-on%20slack-yellow.svg
[license-img]: https://img.shields.io/badge/License-MIT-red.svg
[license-url]: https://opensource.org/licenses/MIT

[lifecycle-img]: https://img.shields.io/badge/lifecycle-stable-blue.svg

This package provides tools to estimate [standard errors](https://en.wikipedia.org/wiki/Standard_error) and [autocorrelation times](https://en.wikipedia.org/wiki/Autocorrelation) of correlated time series. A typical example is a [Markov chain](https://en.wikipedia.org/wiki/Markov_chain) obtained in a [Metropolis Monte Carlo simulation](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm).

**Binning tools:**

* Logarithmic Binning (bins data directly in a logarithmic manner)
  * Size complexity: `O(log(N))`
  * Time complexity: `O(N)`
* Full Binning (keeps all data and allows for any bin size)
* ErrorPropagator (logarithmic binning, can calculate errors of functions of the inputs, experimental)

**Statistical resampling methods:**

* Jackknife resampling (calculates errors for subsamples of the original data)

<br>

As per usual, you can install the registered package with

```julia
] add BinningAnalysis
```

[**OUTDATED** as of v0.6] Note that there is [BinningAnalysisPlots.jl](https://github.com/carstenbauer/BinningAnalysisPlots.jl) which defines some [Plots.jl](https://github.com/JuliaPlots/Plots.jl) recipes for `LogBinner` and `FullBinner` to facilitate visualizing the error convergence.

## Binning tools

All the different binners follow a common interface. 

### Full Binning

The `FullBinner` is a thin wrapper around `Vector` which keeps track of a series of (correlated) data points. To estimate correlation free statistics the data is averaged in bins of variable size. This creates a smaller, less or optimally uncorrelated data set whose statistics are taken in place of the full data set. Since this method features no data compression the bin size can chosen freely.

### Logarithmic Binning

The `LogBinner` is a compressive binning structure. Rather than keeping all the data around, it only keeps data relevant to bin sizes 2^l where l is the binning level. Thus the memory usage drops to `O(log₂(N))` where N is the number of data points, and the choice of bin size becomes a choice of binning level. 

### ErrorPropagator

The `ErrorPropagator` is a derivative of `LogBinner`. It performs logarithmic binning of multiple samples at once, also keeping track of covariances between the samples. Through this errors of functions depending on multiple samples can be derived. The memory complexity of this tool is `O(∏ₖ Dₖ ⋅ log₂(N))` where Dₖ refers to the size of a data point in data set k.

To derive statistics for a function `f` that applies to the samples in `ErrorPropagator` the gradient of `f` at the mean of those values is needed. 

```julia
# Function we want to evaluate
# <x><y> / <xy>
f(v) = v[1] * v[2] / v[3]  

# gradient of f
grad_f(v) = [v[2]/v[3], v[1]/v[3], -v[1]*v[2]/v[3]^2]

# Error propagator with 3 samples
ep = ErrorPropagator(N_args=3)

# Add measurements
x = rand(100)
y = rand(100)
append!(ep, x, y, x .* y)

# calculate mean and error of f
g_mean = mean(ep, f)
Δg = std_error(ep, grad_f)
```

### Interface

You can construct each binner with or without initial data:

```julia
# Create an empty Binner
FB = FullBinner()

# Create a Binner with data
x = rand(100)
LB = LogBinner(x)
```

The element type defaults to `Float64` but can also be adjusted in the constructor. Note that for Array inputs consistent data sizes are assumed and  `LogBinner` and `ErrorPropagator` need a "zero":

```julia
# Create an empty Binner accepting 2x2 Matrices
EP = ErrorPropagator(zeros(2, 2), zeros(2, 2))

# Create a Binner with Complex data
x = rand(ComplexF64, 100)
LB = LogBinner(x)

# Create an empty Binner accepting Complex data
FB = FullBinner(ComplexF64)
```

Further note that `LogBinner` and `ErrorPropagator` have a static capacity which can be set with the `capicity` keyword argument in the constructor. By default the capacity is set to 2^32 (about 4.3 billion values) or the next power of 2 above the `length(data)` when a data set is used to create the binner. Note that you can also create a copy of a binner with a different capacity via `LogBinner(old_binner, capacity = new_capacity)`.

Beyond adding data on creation it can be pushed or appended to a binner:

```julia
FB = FullBinner(ComplexF64)
LB = LogBinner(zeros(3))
EP = ErrorPropagator(Float64, 2)

# Add data
push!(FB, 4.2 + 0.9im)
push!(LB, rand(3))
append!(EP, [1,2,3], [2,3,4])
```

And after all the data has been accumulated we can get statistics of binned data. By default the bin sizes/binning level is picked so that at least 32 bins are included:

```julia
# Get statistics with at least 32 bins
x  = mean(LB)

# Variance
v = var(FB)

# standard error of the mean
Δx = std_error(LB) 

# autocorrelation time
tau_x = tau(FB) 
```

Note that for `ErrorPropagator` you need to specify the data set you want to get statistics from. Alternatively you can also ask for statistics of all included samples by adding an `s` to the functions:

```julia
# mean of first data set
x = mean(EP, 1)

# standard error of the mean for the second
Δy = std_error(EP, 2)

# the autocorrelation time for each data set
tau_xy = taus(EP)
```

If you are interested in a different binning level you can specify it through an optional argument. You can also get statistics for all binning levels through functions with the `all_` prefix:

```julia
# Standard error with binsize 2^(3-1) = 4
Δx4 = std_error(LB, 3)

# autocorrelation time with binsize 7
Δx7 = tau(FB, 7)

# Variance for every binning levels
vs = all_vars(LB)
```

Note that the `all_` functions include all bin sizes from 1 to `div(length(FB), 32)` for a `FullBinner`. For large samples this can be a lot, so it maybe preferable to sample bin sizes manually instead. For `LogBinner` and `ErrorPropagator` there are only `log₂(N)` binsizes so this shouldn't be a problem.


## Resampling methods

The resampling methods currently only include jackknife.

### Jackknife

The jackknife algorithm estimates the mean and standard error of a function applied to a sample `x`. A general k-jackknife does this by creating sub-samples with N-k values, applying the function to the means of these sub-samples and calculating statistics of the results. Our implementation uses `k = 1` and allows for any number of samples:

```julia
x = rand(100)

# jackknife estimates for mean and standard error of <x>
xmean, Δx = jackknife(identity, x)

# in this example
# isapprox(Δx, std(x)/sqrt(length(x))) == true

# jackknife estimates for mean and standard error of <1/x>
x_inv_mean, Δx_inv = jackknife(identity, 1 ./ x) 

# Multiple time series
x = rand(100)
y = rand(100)

# The inputs of the function `g` must be provided as arguments in `jackknife`.
g(x, y, xy) = x * y / xy  # <x><y> / <xy>
g_mean, Δg = jackknife(g, x, y, x .* y)
```

### Error Propagator

```julia
ep = ErrorPropagator(N_args=1)
# Essentially a LogBinner that can hold multiple variables. Errors can be derived
# for functions which depend on these variables. The memory overhead of this
# type is O(N_args^2 log(N_samples)), making it much cheaper than jackknife for
# few variables

push!(ep, rand())
append!(ep, rand(99))

# Mean and error of the (first) input
xmean = mean(ep, 1)
Δx = std_error(ep, 1)

# To compute the mean and error of a function we need its gradient
f(x) = x.^2
dfdx(x) = 2x
y = mean(ep, f)[1]
Δy = std_error(ep, dfdx)[1]

# Error propagator with multiple variables:
ep = ErrorPropagator(N_args=3)

# Multiple time series
x = rand(100)
y = rand(100)
append!(ep, x, y, x.*y)

# means and standard error of inputs:
xs = means(ep)
Δxs = std_errors(ep)

# mean and error of a function dependant on x, y and xy
# Note that this function takes a vector input
g(v) = v[1] * v[2] / v[3]  # <x><y> / <xy>
dgdx(v) = [v[2]/v[3], v[1]/v[3], -v[1]*v[2]/v[3]^2]
g_mean = mean(ep, g)
Δg = std_error(ep, dgdx)
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
