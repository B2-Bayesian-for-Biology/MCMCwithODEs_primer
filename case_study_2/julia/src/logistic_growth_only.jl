# For user-defined post processing and plotting functions
include(joinpath(@__DIR__, "..", "..", "..", "utils", "plot_utils.jl"))

## Cell 1 ## 

using CSV, DataFrames

# from your data-prep cell
df = CSV.read("../../../case_study_2/python/data/total_cells.csv", DataFrame)

times    = df[end-14:end, :1]
y_obs    = df[end-14:end, :2] * 1e6
log_y_obs = log.(y_obs .+ 1e-9)

## Cell 2 ##

function ode(du, u, p, t)
    P = u
    r, K = p

    du[1] = r * (1 - P / K) * P

    return nothing
end

## Cell 3 ##

using Turing
@model function fit_ode(log_y_obs, times, prob)
    r ~ Uniform(0.5, 1.0)
    K ~ Uniform(1e6, 4e7)

    P0 ~ Uniform(1e5, 3e5)
    
    sigma ~ truncated(Normal(0, 3), 0, Inf)

    pr = remake(prob;
                p = [r, K],
                u0 = [P0],
                tspan = (times[1], times[end]))

    sol = solve(pr, Tsit5();
                abstol = 1e-6, reltol = 1e-6,
                saveat = times)

    log_y_pred = log.(Array(sol)[1, :] .+ 1e-9)
    log_y_obs ~ arraydist(Normal.(log_y_pred, sigma))
end

## Cell 4 ##

using DifferentialEquations

u0 = [y_obs[1]]
p = [0.5, 1e6]
tspan = (times[1], times[end])
prob = ODEProblem(ode, u0, tspan, p)

model    = fit_ode(log_y_obs, times, prob)
chain    = sample(model, NUTS(1000, .95), MCMCSerial(), 1000, 4; progress=false)

priors = Dict{Symbol,Distribution}(
    :r     => Uniform(0.5, 1.0),
    :K     => Uniform(1e6, 4e7),
    :P0    => Uniform(1e5, 3e5),
    :sigma => truncated(Normal(0, 3), 0, Inf),
)

order = [:r, :K, :P0, :sigma]
plot_trace_with_priors(chain; priors=priors, var_order=order, per_chain_density=true)  # also per-chain densities

init_syms = [:P0]
param_syms = [:r, :K]
t_obs = times
y_obs = y_obs

plt = overlay_posterior_on_observed(
    chain, ode, t_obs, y_obs;
    init_syms=init_syms,
    param_syms=param_syms,
    which_states=[1],     # choose states to plot
    n_draws=150,            # how many posterior paths to overlay
    plot_ribbon=true,       # median ± CI band
    legend=:topleft,
    ribbon_q=(0.1, 0.9),    # CI limits
    logy=false
)
display(plt)