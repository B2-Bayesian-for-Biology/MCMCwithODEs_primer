using StatsPlots, Distributions, MCMCChains, Measures

function plot_trace_with_priors(
    chain::Chains;
    priors::Dict{Symbol,Distribution},
    var_order::Union{Nothing,Vector{Symbol}}=nothing,
    var_labels::Union{Nothing,Dict{Symbol,AbstractString}}=nothing,
    nxs::Int=500, per_chain_density::Bool=false,
    legend=false,
    tickfontsize::Int=6, guidefontsize::Int=8, legendfontsize::Int=9,
    margin_mm::Real=14, extra_right_mm::Real=10, top_mm::Real=6, bottom_mm::Real=8
)
    # fonts
    default(tickfont=font(tickfontsize),
            guidefont=font(guidefontsize),
            legendfont=font(legendfontsize))

    # order of parameters
    ps = var_order === nothing ? collect(keys(priors)) : copy(var_order)
    rows = length(ps)

    # figure
    plt = plot(layout=(rows, 2), size=(1000, 240*rows),
               left_margin=margin_mm*mm, right_margin=(margin_mm+extra_right_mm)*mm,
               top_margin=top_mm*mm, bottom_margin=bottom_mm*mm)

    # helper to get a nice title for each param
    label_for = p -> (var_labels === nothing ? string(p) : get(var_labels, p, string(p)))

    for (i, p) in enumerate(ps)
        @assert haskey(priors, p) "Missing prior for parameter $(p)."

        A = Array(chain[p])                 # [iters x nchains]
        iters, nch = size(A)

        # Left: traces (all chains)
        plot!(plt[i, 1],
              title = label_for(p),
              xlabel="Iteration", ylabel="Sample value",
              legend=legend, 
              framestyle=:box)
        for c in 1:nch
            plot!(plt[i, 1], 1:iters, A[:, c], label="chain $c")
        end

        # Right: posterior density (+ optional per-chain) + prior overlay
        samp = vec(A)
        density!(plt[i, 2], samp;
                 label="Posterior",
                 xlabel="Sample value", ylabel="Density",
                 title=label_for(p),
                 legend=legend, framestyle=:box)

        if per_chain_density
            for c in 1:nch
                density!(plt[i, 2], A[:, c]; label="chain $c", alpha=0.45)
            end
        end

        prior = priors[p]
        lo = min(quantile(prior, 0.001), minimum(samp))
        hi = max(quantile(prior, 0.999), maximum(samp))
        if !(hi > lo)  # guard against zero range
            lo -= 1; hi += 1
        end
        xs = range(lo, hi; length=nxs)
        plot!(plt[i, 2], xs, pdf.(prior, xs);
              lw=2, ls=:dash, color=:black, label="Prior")

        # small extra x padding prevents clipped tick labels/exponent
        rng = hi - lo
        plot!(plt[i, 2]; xlims=(lo - 0.04*rng, hi + 0.08*rng))
    end

    return plt
end

#################################################






# Overlay posterior predicted trajectories on observed data (Julia + Turing + DifferentialEquations)
#
# What this does
# - Draws N posterior samples from an MCMCChains.Chains object
# - For each draw, builds an ODEProblem with u0 and parameters from that draw
# - Solves the ODE at your observation times
# - Plots many predicted trajectories as faint lines
# - Overlays observed data as markers
# - Optionally adds a credible ribbon (pointwise quantiles across draws)
# - Supports 1 or multiple state variables (choose which_state(s) to plot)
#
# Minimal deps: StatsPlots, MCMCChains, DifferentialEquations

using StatsPlots
using MCMCChains
using DifferentialEquations
using Statistics

# --- Helper: extract a flat Vector of sampled values for a parameter symbol ---
_paramvec(chain::Chains, s::Symbol) = vec(Array(chain[s]))

# --- Main function ---

function overlay_posterior_on_observed(
    chain::Chains,
    f!::F,
    t_obs::AbstractVector,
    y_obs::Union{AbstractVector,AbstractMatrix};
    init_syms::Vector{Symbol},
    param_syms::Vector{Symbol},
    which_states::Union{Int,AbstractVector{Int}}=1,
    pred_transforms::Union{Nothing,Vector{Function}}=nothing,
    n_draws::Int=200,
    solver=Tsit5(),
    abstol=1e-8, reltol=1e-8,
    plot_ribbon::Bool=true,
    ribbon_q=(0.05, 0.95),
    alpha_paths=0.15,
    lw_paths=1.2,
    markersize=5,
    ms_alpha=0.9,
    legend=:topright,
    title_suffix::AbstractString="",
    logy::Bool=false,
) where {F<:Function}

    T = length(t_obs)
    ismatrix = y_obs isa AbstractMatrix

    # Normalize which_states to a Vector{Int}
    ws = isa(which_states, Int) ? [which_states] : collect(which_states)

    # How many posterior draws are available
    nsamp_total = length(_paramvec(chain, param_syms[1]))
    n_paths = min(n_draws, nsamp_total)
    draw_idx = round.(Int, range(1, nsamp_total; length=n_paths))

    # 1) Simulate selected raw states for each posterior draw
    #    Ystates has shape: T × length(ws) × n_paths
    Ystates = Array{Float64}(undef, T, length(ws), n_paths)

    tspan = (first(t_obs), last(t_obs))
    for (j, k) in enumerate(draw_idx)
        u0 = [ _paramvec(chain, s)[k] for s in init_syms ]
        p  = [ _paramvec(chain, s)[k] for s in param_syms ]
        prob = ODEProblem(f!, u0, tspan, p)
        sol = solve(prob, solver; abstol=abstol, reltol=reltol, saveat=t_obs)
        for (ii, sidx) in enumerate(ws)
            @inbounds Ystates[:, ii, j] = sol[sidx, :]
        end
    end

    # 2) Build observables to plot. If no transforms, use raw states.
    #    Yplot has shape: T × n_pan × n_paths
    if pred_transforms === nothing
        Yplot = Ystates
        n_pan = length(ws)
    else
        m = length(pred_transforms)
        n_pan = m
        Yplot = Array{Float64}(undef, T, m, n_paths)
        @inbounds for j in 1:n_paths
            M = @view Ystates[:, :, j]  # T × length(ws)
            for c in 1:m
                f = pred_transforms[c]   # f takes Vector of current state values
                for t in 1:T
                    Yplot[t, c, j] = f(vec(@view M[t, :]))
                end
            end
        end
    end

    # 3) Plot
    plt = plot(layout=(n_pan, 1), size=(900, max(260, 260*n_pan)), legend=legend,
               left_margin=10mm, right_margin=10mm, top_margin=6mm, bottom_margin=6mm)

    for ii in 1:n_pan
        # Posterior paths
        for j in 1:n_paths
            plot!(plt[ii], t_obs, @view Yplot[:, ii, j]; color=:gray, alpha=alpha_paths, lw=lw_paths,
                  label=(j==1 ? "Posterior draws" : ""))
        end

        # Credible ribbon (avoid dropped-dims error by working with T×n_paths directly)
        if plot_ribbon
            A = @view Yplot[:, ii, :]                   # T×n_paths

            ql(q) = mapslices(x -> quantile(x, q), A; dims=2)[:]
            lo  = ql(ribbon_q[1])
            hi  = ql(ribbon_q[2])
            med = ql(0.5)

            plot!(plt[ii], t_obs, med; ribbon=(med .- lo, hi .- med), label="median ± CI", lw=2)
        end

        # Observed data overlay
        if ismatrix
            # If transforms are provided, columns of y_obs correspond to those transforms;
            # otherwise, columns correspond to the selected states in order.
            @assert size(y_obs, 1) == T "y_obs rows must match t_obs length"
            @assert size(y_obs, 2) >= ii "y_obs must have a column for panel $(ii)"
            obs_series = @view y_obs[:, ii]
            scatter!(plt[ii], t_obs, obs_series; ms=markersize, msalpha=ms_alpha,
                     label="observed", marker=:circle)
        else
            @assert n_pan == 1 "y_obs is a vector; use a matrix for multiple observables."
            scatter!(plt[ii], t_obs, y_obs; ms=markersize, msalpha=ms_alpha,
                     label="observed", marker=:circle)
        end

        # Axes / titles
        plot!(plt[ii]; framestyle=:box)
        xlabel!(plt[ii], "time")
        ylabel!(plt[ii], (pred_transforms === nothing) ? "state $(ws[ii])" : "observable $(ii)")
        if logy
            plot!(plt[ii]; yscale=:log10)
        end

        title_txt = (pred_transforms === nothing && n_pan == 1) ?
                    "Posterior predictions vs observed" :
                    (pred_transforms === nothing ? "State $(ws[ii]): predictions vs observed" : "Observable $(ii): predictions vs observed")
        if !isempty(title_suffix)
            title_txt *= " — " * title_suffix
        end
        title!(plt[ii], title_txt)
        xlims!(plt[ii], (first(t_obs), last(t_obs)))
    end

    return plt
end

# ----------------------------------------
# Example usage (edit these to your model)
# ----------------------------------------
# Suppose your model is two-state (e.g., u = [P, D]) with params θ = [r, K, δ]
# and u0 = [P0, D0]. Replace with your actual f! and data.

# Example RHS (placeholder) — replace with your ODE
# function f!(du, u, p, t)
#     r, K, δ = p
#     P, D = u
#     du[1] = r*P*(1 - P/K) - δ*P      # dP/dt
#     du[2] = δ*P                       # dD/dt
# end

# t_obs = collect(0.0:1.0:20.0)               # your observation times
# y_obs = [/* T×2 matrix of observed [P D] */] # e.g., from a DataFrame

# init_syms  = [:P0, :D0]
# param_syms = [:r, :K, :delta]

# plt = overlay_posterior_on_observed(
#     chain,
#     f!,
#     t_obs,
#     y_obs;
#     init_syms=init_syms,
#     param_syms=param_syms,
#     which_states=[1,2],        # plot both P and D panels
#     n_draws=150,
#     plot_ribbon=true,
#     ribbon_q=(0.1, 0.9),
#     logy=false,
#     title_suffix="Turing posterior"
# )
# display(plt)

# Notes
# - If you only observe P (state 1), pass y_obs as a Vector for P and set which_states=1.
# - If your observation model adds noise (e.g., Normal(P, σ)), you typically plot the *latent* P
#   trajectories (as above) against the observed points; optionally add error bars to y_obs.
# - For stiff ODEs, consider Rosenbrock23() or other stiff solvers.
# - If your t_obs are not strictly increasing, sort them or use unique(t_obs) with care.



using DataFrames, StatsBase, DifferentialEquations, Plots

"""
Posterior trajectories for:
  - Total cells = P + D
  - Dead cells  = D
  - Nutrients   = N

Matches reference layout: Total + Dead on log scale, Nutrients linear.
"""
function plot_posterior_states_aligned_case_3(chain, prob, times;
    n_draws::Int=200,
    ribbon_q::Tuple{<:Real,<:Real}=(0.05, 0.95),
    solver=Rodas5(), abstol=1e-6, reltol=1e-6,
    alpha_paths=0.08, lw_paths=1.0,
    obs_total=nothing,
    obs_dead=nothing
)

    df = DataFrame(chain)
    required = [:mum, :Ks, :Qn, :delta, :P0, :D0]
    miss = setdiff(required, propertynames(df))
    !isempty(miss) && error("Chain is missing parameters: $(miss)")

    N0_from_Qn(qn) = 1000 + ((500 / 1.8e-10) * (qn - 3.2e-10))
    n = nrow(df)
    idxs = rand(1:n, min(n_draws, n))
    T = length(times)

    total_paths = Matrix{Float64}(undef, T, length(idxs))
    dead_paths  = Matrix{Float64}(undef, T, length(idxs))
    nutr_paths  = Matrix{Float64}(undef, T, length(idxs))

    for (j, i) in enumerate(idxs)
        mum = df.mum[i]; Ks = df.Ks[i]; Qn = df.Qn[i]; δ = df.delta[i]
        P0  = df.P0[i];  D0 = df.D0[i]
        N0  = hasproperty(df, :N0) ? df.N0[i] : N0_from_Qn(Qn)

        pr  = remake(prob; u0=[N0, P0, D0], p=[mum, Ks, Qn, δ], tspan=(times[1], times[end]))
        sol = solve(pr, solver; saveat=times, abstol=abstol, reltol=reltol)

        N = sol[1, :]
        P = sol[2, :]
        D = sol[3, :]

        nutr_paths[:, j]  = max.(N, 0.0)
        dead_paths[:, j]  = max.(D, 0.0)
        total_paths[:, j] = max.(P .+ D, 0.0)
    end

    quant_band(M) = (mapslices(x -> quantile(x, ribbon_q[1]), M; dims=2)[:],
                     mapslices(x -> quantile(x, ribbon_q[2]), M; dims=2)[:])

    loT, hiT = quant_band(total_paths)
    loD, hiD = quant_band(dead_paths)
    loN, hiN = quant_band(nutr_paths)

    # Total (log scale)
    p1 = plot(times, total_paths; color=:steelblue, alpha=alpha_paths, lw=lw_paths, label="",
        yscale=:log10, xlabel="Time (days)", ylabel="Density", title="Total cells (/ml)")
    plot!(times, loT, ribbon=(hiT .- loT), lw=2, fillalpha=0.0, label="Credible band")
    if obs_total !== nothing
        scatter!(times, obs_total; ms=5, color=:black, label="", alpha=0.9)
    end

    # Dead (log scale)
    p2 = plot(times, dead_paths; color=:steelblue, alpha=alpha_paths, lw=lw_paths, label="",
        yscale=:log10, xlabel="Time (days)", ylabel="Density", title="Dead cells (/ml)")
    plot!(times, loD, ribbon=(hiD .- loD), lw=2, fillalpha=0.0, label="Credible band")
    if obs_dead !== nothing
        scatter!(times, obs_dead; ms=5, color=:black, label="", alpha=0.9)
    end

    # Nutrients (linear scale)
    p3 = plot(times, nutr_paths; color=:steelblue, alpha=alpha_paths, lw=lw_paths, label="",
        xlabel="Time (days)", ylabel="mmol N m⁻³", title="Nutrients")
    plot!(times, loN, ribbon=(hiN .- loN), lw=2, fillalpha=0.0, label="Credible band")

    plot(p1, p2, p3; layout=(1,3), size=(1100, 350), margin=5mm)
end







using DataFrames, StatsBase, DifferentialEquations, Plots

function plot_posterior_states_stacked(chain, prob, times;
    n_draws::Int=200,
    ribbon_q=(0.05, 0.95),              # CI (e.g., 90%)
    solver=Rodas5(), abstol=1e-6, reltol=1e-6,
    alpha_draws=0.10, lw_draws=1.0,     # faint grey draws
    ci_color=:deeppink3, ci_alpha=0.25, # pink median + CI
    obs_total=nothing, obs_dead=nothing,
    size=(900, 1050)                    # tall for vertical stack
)
    df = DataFrame(chain)
    required = [:mum, :Ks, :Qn, :delta, :P0, :D0]
    miss = setdiff(required, propertynames(df))
    !isempty(miss) && error("Chain is missing parameters: $(miss)")

    N0_from_Qn(qn) = 1000 + ((500 / 1.8e-10) * (qn - 3.2e-10))
    n = nrow(df); idxs = rand(1:n, min(n_draws, n))
    T = length(times)

    total_paths = Matrix{Float64}(undef, T, length(idxs))
    dead_paths  = Matrix{Float64}(undef, T, length(idxs))
    nutr_paths  = Matrix{Float64}(undef, T, length(idxs))

    for (j, i) in enumerate(idxs)
        mum=df.mum[i]; Ks=df.Ks[i]; Qn=df.Qn[i]; δ=df.delta[i]
        P0=df.P0[i];  D0=df.D0[i]
        N0 = hasproperty(df,:N0) ? df.N0[i] : N0_from_Qn(Qn)

        pr  = remake(prob; u0=[N0,P0,D0], p=[mum,Ks,Qn,δ], tspan=(times[1], times[end]))
        sol = solve(pr, solver; saveat=times, abstol=abstol, reltol=reltol)

        N = sol[1, :]; P = sol[2, :]; D = sol[3, :]
        nutr_paths[:, j]  = max.(N, 0.0)
        dead_paths[:, j]  = max.(D, 0.0)
        total_paths[:, j] = max.(P .+ D, 0.0)
    end

    # helpers
    quant_band(M, q1, q2) = (mapslices(x->quantile(x,q1), M; dims=2)[:],
                             mapslices(x->quantile(x,0.5), M; dims=2)[:],
                             mapslices(x->quantile(x,q2), M; dims=2)[:])

    loT, medT, hiT = quant_band(total_paths, ribbon_q[1], ribbon_q[2])
    loD, medD, hiD = quant_band(dead_paths,  ribbon_q[1], ribbon_q[2])
    loN, medN, hiN = quant_band(nutr_paths,  ribbon_q[1], ribbon_q[2])

    # style for draws
    draw_kwargs = (color=:gray, alpha=alpha_draws, lw=lw_draws, label=false)

    # 1) Total (log)
    p1 = plot(times, total_paths; draw_kwargs..., yscale=:log10,
              xlabel="time", ylabel="observable 1", title="Observable 1: predictions vs observed",
              legend=:topleft)
    plot!(times, medT; color=ci_color, lw=2, label="median ± CI")
    plot!(times, medT; ribbon=(medT .- loT, hiT .- medT), fillalpha=ci_alpha, fillcolor=ci_color, label="")
    if obs_total !== nothing
        scatter!(times, obs_total; color=:orange, ms=5, markerstrokecolor=:black, label="observed")
    end

    # 2) Dead (log)
    p2 = plot(times, dead_paths; draw_kwargs..., yscale=:log10,
              xlabel="time", ylabel="observable 2", title="Observable 2: predictions vs observed",
              legend=:topleft)
    plot!(times, medD; color=ci_color, lw=2, label="median ± CI")
    plot!(times, medD; ribbon=(medD .- loD, hiD .- medD), fillalpha=ci_alpha, fillcolor=ci_color, label="")
    if obs_dead !== nothing
        scatter!(times, obs_dead; color=:orange, ms=5, markerstrokecolor=:black, label="observed")
    end

    # 3) Nutrients (linear)
    p3 = plot(times, nutr_paths; draw_kwargs..., xlabel="time", ylabel="mmol N m⁻³",
              title="Nutrients: predictions", legend=:topleft)
    plot!(times, medN; color=ci_color, lw=2, label="median ± CI")
    plot!(times, medN; ribbon=(medN .- loN, hiN .- medN), fillalpha=ci_alpha, fillcolor=ci_color, label="")

    plot(p1, p2, p3; layout=(3,1), size=size, margin=8mm, left_margin=10mm, bottom_margin=8mm)
end
