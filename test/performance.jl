using PyPlot, BinningAnalysis

function performance_test(N = 28, start=1)
    timings = Float64[]

    for i in start:N
        # NOTE using the default level here actually performs better
        BA = BinnerA(i+1)
        xs = rand(2^i)
        GC.gc()

        t0 = time_ns()
        for x in xs
            push!(BA, x)
        end
        t = time_ns() - t0

        println("2^$i done in $(t)ns.")
        push!(timings, t / length(xs))
    end

    start:N, timings
end

performance_test(1)
xs, ys = performance_test(26)

begin
    figure()
    plot(xs, ys, "r+:")
    xlabel("log(Values pushed)")
    ylabel("Average time per push [ns]")
    ylim(0, 100)
end


# Memory: 33B * N where N ≥ ⌈log(K+1)⌉ with K values pushed
# Time:   10-40ns per push, depending on luck and how I count
