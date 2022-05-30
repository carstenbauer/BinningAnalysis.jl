This documents contains notes on the autocorrelations returned by different methods.

# Methods used

BinningAnalysis.jl implements 3 user facing tools for estimating autocorrelation times - `FullBinner`, `LogBinner` and `ErrorPropagator`. The latter two are equivalent in terms of calculating autocorrelation times, so we will ignore `ErrorPropagator` here.

The packages also implements another estimate for autocorrelation times which is currently not exported. Based on the book Quantum Monte Carlo Methods, Chapter 3.3 we have `BinningAnalysis.unbinned_tau(sample)` implementing

$$
\tau = \sum_{k = 1}^{N - 1} (1 - \frac{k}{M}) \frac{\chi_k}{\chi_0}
$$

as another estimate for the autocorrelation time and `BinningAnalysis.correlation(sample, k)` implementing

$$
\chi_k = \langle x_i x_{i+k} \rangle - \langle x_i \rangle \langle x_{i+k} \rangle
$$

as a measure for correlation between two values x in the sample.

We note that `BinningAnalysis.unbinned_tau` truncates the sum over $\chi_k$ once those values becomes relatively small. This is done to speed up the calculation and cut off a long tail of correlations oscillating around 0.

# Integration Tests

In the following sections we will take a look at the tests in "test/systematic_correlations.jl". These tests aim to verify that we can correctly asses (or at least approximate) the autocorrelation times and statistics of a few more or less engineered samples.

## Correlated blocks of random values

For this test we engineer a sample with very obvious correlations - repeated values. We construct the correlated sample as

```julia
N_corr = 16
N_blocks = 131_072 # 2^17

uncorrelated = rand(N_blocks)    
correlated = [x for x in uncorrelated for _ in 1:N_corr]

using GLMakie

fig, ax, p = scatter(correlated[1:300], markersize = 4, strokewidth=1, color = :lightblue)
xlims!(ax, 0, 200)
ylims!(ax, 0, 1)
ax.xlabel[] = "Sample Index"
ax.ylabel[] = "Sample value"
fig
```

![Sample distribution](https://github.com/carstenbauer/BinningAnalysis.jl/blob/ff/cleanup/docs/src/assets/notes/blocks_sample.png)

With this setup we are guaranteed to hit a new uncorrelated value whenever we shift the current index by $\pm 16$. On average we will find a new uncorrelated value whenever we shift by $k = \pm 8$. So what is our autocorrelation time in this case? With the definitions from the Quantum Monte Carlo book it seems to be 7.5 - i.e. the mid point between "correlated on average" and "uncorrelated on average".

### Correlation Function and `unbinned_tau`

With 7.5 set as a target for the autocorrelation time, let us look at the different tools BinningAnalysis provides. First of we look at the most analytical approach - the `correlation` function and `unbinned_tau`. We expect the correlations to decrease with increasing distance k up to `k = N_corr`, where we are guaranteed to hit a new random value.

```julia
using BinningAnalysis
using BinningAnalysis: correlation, unbinned_tau

chis = [correlation(correlated, k) for k in 1:100]
ks_full = rand(1:length(correlated)-32, 200)
chis_full = [correlation(correlated, k) for k in ks_full]

fig = Figure(resolution = (1200, 500))
left = Axis(fig[1, 1], xlabel = "Distance k", ylabel = "Correlation χ(k)")
scatter!(left, chis, color = :lightblue, markersize = 5, strokewidth = 1)
vlines!(left, N_corr, color = :red, linestyle = :dash)

right = Axis(fig[1, 2], xlabel = "Distance k", ylabel = "Correlation χ(k)")
scatter!(right, ks_full, chis_full, color = :lightblue, markersize = 5, strokewidth = 1)
ylims!(right, -0.011, 0.011)

fig
```

![Correlations](https://github.com/carstenbauer/BinningAnalysis.jl/blob/ff/cleanup/docs/src/assets/notes/blocks_correlations.png)

In the left plot we can nicely see how the correlations decrease up to the red line, indicating a distance of `N_corr`. Beyond that we get a long tail of values close to 0. As we see in the right plot however these values are only about 100 times smaller than the largest correlation at k = 1. When calculating the autocorrelation time this may push us to different depending on the number of points we include. Here are some example values for $\tau$ following from these correlations:

- `k = 1:16` yields $\tau \approx 7.477$
- `k = 1:32` yields $\tau \approx 7.443$
- `k = 1:100` yields $\tau \approx 7.372$
- `k = 1:1000` yields $\tau \approx 7.333$
- `k = 1:10000` yields $\tau \approx 7.763$
- `unbinned_tau(correlated)` yields $\tau \approx 7.48$


### FullBinner


Next let us look at `FullBinner` where we can set an arbitrary bin size. The autocorrelation time we measure can not exceed `0.5(binsize - 1)` so we expect to see these values up to `binsize = N_corr`. Beyond that the autocorrelation time should oscillate around 7.5.

```julia
FB = FullBinner(correlated)

_taus = [tau(FB, binsize) for binsize in 1:100]
binsizes_full = rand(1:div(length(FB), 32), 200)
_taus_full = [tau(FB, binsize) for binsize in binsizes_full]

fig = Figure(resolution = (1200, 500))
left = Axis(fig[1, 1], xlabel = "Binsize", ylabel = "Autocorrelation time τ")
scatter!(left, _taus, color = :lightblue, markersize = 5, strokewidth = 1)
vlines!(left, N_corr, color = :red, linestyle = :dash)
hlines!(left, 7.5, color = :red, linestyle = :dash)
lines!(left, 1:32, 0.5 .* (0:31), color = :orange, linestyle = :dash)
ylims!(left, -0.5, 10.5)

right = Axis(fig[1, 2], xlabel = "Binsize", ylabel = "Autocorrelation time τ")
scatter!(right, binsizes_full, _taus_full, color = :lightblue, markersize = 5, strokewidth = 1)
hlines!(right, 7.5, color = :red, linestyle = :dash)
hlines!(right, mean(_taus_full), color = :blue)
ylims!(right, -0.5, 10.5)

fig
```

![FullBinner Autocorrelation time](https://github.com/carstenbauer/BinningAnalysis.jl/blob/ff/cleanup/docs/src/assets/notes/blocks_FB_tau.png)

This plot shows a bunch of odd features: we don't follow `0.5(binsize - 1)` for all points before `binsize = N_corr` and afterwards only some points agree with $\tau = 7.5$. Both of these can be explained by comparing the binsize with `N_corr`.

Let us use `binsize = N_corr + 1` as an example. With this we will always have a value from another correlated block in our bin. This extra value is obviously correlated to the rest of its block, which messes with the estimation of the autocorrelation time. On the flip side, every time our bin size is a multiple of `N_corr` we have no correlation between different bins. Similarly if the bin size is a divisor of `N_corr` we have a clean number of correlated bins.

In the right plot we see a significant amount of variance which is likely just a result of the low number of bins (`binsize = 1e4` results in 209 bins). The average autocorrelation time (blue line) is 7.885 in this case which is reasonably close.


### LogBinner


The LogBinner only includes bin sizes $2^l$, i.e. bin sizes which are compatible with the `N_corr` we picked. Thus we should generally see well fitting values.

```julia
LB = LogBinner(correlated)

_taus = all_taus(LB)
cutoff = BinningAnalysis._reliable_level(LB)

fig = Figure(resolution = (600, 500))
ax = Axis(fig[1, 1], xlabel = "Logarithmic binning level l", ylabel = "Autocorrelation time τ")
scatter!(ax, 0:length(_taus)-1, _taus, color = :lightblue, markersize = 5, strokewidth = 1)
vlines!(ax, [4, cutoff], color = [:red, :black], linestyle = :dash)
hlines!(ax, 7.5, color = :red, linestyle = :dash)
lines!(ax, 0:4, 0.5 .* (2 .^(0:4) .- 1), color = :orange)
ylims!(ax, -0.5, 10.5)

fig
```

![LogBinner Autocorrelation time](https://github.com/carstenbauer/BinningAnalysis.jl/blob/ff/cleanup/docs/src/assets/notes/blocks_LB_tau.png)

In this case the values for the autocrrelation time nicely match up with `0.5(binsize-1)` (orange line) up to `binsize = N_corr = 2^4` where they saturate at 7.5 (red dashed lines). Beyond that the autocorrelation time remains stable for a while and then starts varying due to statistic fluctuations. BinningAnalysis sets the cutoff for those at 32 bins, which is indicated by the black dashed line.

Of course this nice correspondence is engineered by setting `N_corr` to a power of 2. If we set it to 15, for example, we get this:

![LogBinner Autocorrelation time 15](https://github.com/carstenbauer/BinningAnalysis.jl/blob/ff/cleanup/docs/src/assets/notes/blocks_LB_tau15.png)

The initial growth no longer matches up and the plateau at 7.0 settles in quite a bit later, but we can still easily find the correct autocorrelation time. 

## Real data

Next let us look at some data generated from a Monte Carlo simulation of the Ising model. (More specifically these are energies from an Ising model with nearest neighbor hopping on a triangular lattice. The linear system size is L = 8 and the temperature T = 2.)

```julia
data = open(joinpath(pkgdir(BinningAnalysis), "test/test.data"), "r") do f
    Float64[read(f, Float64) for _ in 1:2^14]
end


FB = FullBinner(data)
LB = LogBinner(data)

x = correlation(data, 0)
ys = map(k -> (1 - k/length(data)) * correlation(data, k) / x, 1:2^14-32)
_taus = cumsum(ys)

fig = Figure(resolution = (1200, 400))
ax1 = Axis(fig[1, 1], xlabel = "cutoff", ylabel = "Autocorrelation time τ")
scatter!(ax1, _taus, color = :lightblue, markersize = 4, strokewidth = 1)

ax2 = Axis(fig[1, 2], xlabel = "Bin size", ylabel = "Autocorrelation time τ")
scatter!(ax2, all_taus(FB), color = :lightblue, markersize = 4, strokewidth = 1)

ax3 = Axis(fig[1, 3], xlabel = "Bin size", ylabel = "Autocorrelation time τ")
scatter!(ax3, all_taus(LB), color = :lightblue, markersize = 4, strokewidth = 1)

ax1.title[] = "Unbinned"
ax2.title[] = "FullBinner"
ax3.title[] = "LogBinner"
for ax in (ax1, ax2, ax3)
    ylims!(ax, 0.0, 14.0)
    hlines!(ax, 4.0, color = :black, linestyle = :dash)
end

fig
```

![Autocorrelation times](https://github.com/carstenbauer/BinningAnalysis.jl/blob/ff/cleanup/docs/src/assets/notes/data_combined_tau.png)

In this example both binning methods agree on a autocorrelation time of roughly 4. The unbinned method on the other hand does not find a stable value. Let us look at the correlations in more detail.

```julia
ys = [correlation(data, k) for k in 1:2^14-32]

fig = Figure(resolution = (900, 400))
ax1 = Axis(fig[1, 1], xlabel = "Distance k", ylabel = "Correlation χ(k)")
scatter!(ax1, ys, color = :lightblue, markersize = 4, strokewidth = 1)
xlims!(ax1, -1, 100)

ax2 = Axis(fig[1, 2], xlabel = "Distance k", ylabel = "Correlation χ(k)")
scatter!(ax2, ys, color = :lightblue, markersize = 4, strokewidth = 1)
xlims!(ax2, -3e2, 1.7e4)
fig
```

![Correlations](https://github.com/carstenbauer/BinningAnalysis.jl/blob/ff/cleanup/docs/src/assets/notes/data_corelations.png)

Like in the previous example the correlations start of large and decay, but the variance is larger than before. Furthermore the correlations don't average to 0 after the initial decay, resulting an increasing autocorrelation. If we cut off the summation early on we find a similar autocorrelation time as in the binned methods:

- $k_{max} = 25$ yields $\tau \approx 3.80$
- $k_{max} = 50$ yields $\tau \approx 3.99$
- $k_{max} = 75$ yields $\tau \approx 4.19$
- $k_{max} = 100$ yields $\tau \approx 4.41$