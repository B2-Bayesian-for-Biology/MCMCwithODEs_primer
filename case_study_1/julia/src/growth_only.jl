# For user-defined post processing and plotting functions
include(joinpath(@__DIR__, "..", "..", "..", "utils", "plot_utils.jl"))

## Cell 1 ##

using CSV, DataFrames

df = CSV.read("../../../case_study_1/python/data/phaeocystis_control.csv", DataFrame)

times    = df.times
y_obs    = df.cells
log_y_obs = log.(y_obs .+ 1e-9)

## Cell 2 ##

function ode(du, u, p, t)
    mum = p[1]
    y = u[1]
    du[1] = mum * y
    return nothing
end

## Cell 3 ##

using Turing
@model function fit_ode(log_y_obs, times, prob)
    mum   ~ truncated(Normal(0.5, 0.3), 0.0, 1.0)
    N0    ~ LogNormal(log(1_630_000), 0.1)
    sigma ~ truncated(Normal(0, 1), 0, Inf)    
    
    pr = remake(prob;
                p = [mum],
                u0 = [N0],
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
p = [0.5]
tspan = (times[1], times[end])
prob = ODEProblem(ode, u0, tspan, p)

model = fit_ode(log_y_obs, times, prob)
chain = sample(model, NUTS(1000, .95), MCMCSerial(), 1000, 3; progress=false)

priors = Dict{Symbol,Distribution}(
    :mum   => truncated(Normal(0.5, 0.3), 0.0, 1.0),
    :N0    => LogNormal(log(1_630_000.0), 0.1),
    :sigma => truncated(Normal(0, 1.0), 0.0, Inf)
)

order = [:mum, :N0, :sigma]
plot_trace_with_priors(chain; priors=priors, var_order=order, per_chain_density=true)  # also per-chain densities

init_syms = [:N0]
param_syms = [:mum]
t_obs = times
y_obs = y_obs

plt = overlay_posterior_on_observed(
    chain, ode, t_obs, y_obs;
    init_syms=init_syms,
    param_syms=param_syms,
    which_states=[1],     # choose states to plot
    n_draws=150,            # how many posterior paths to overlay
    plot_ribbon=true,       # median ± CI band
    ribbon_q=(0.1, 0.9),    # CI limits
    legend=:topleft,
    logy=false
)
display(plt)

