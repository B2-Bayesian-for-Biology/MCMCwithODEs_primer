
# For user-defined post processing and plotting functions
include(joinpath(@__DIR__, "..", "..", "..", "utils", "plot_utils.jl"))

using CSV, DataFrames

df = CSV.read("../data/phaeocystis_control.csv", DataFrame)

times    = df.times
y_obs    = df.cells
log_y_obs = log.(y_obs .+ 1e-9)

function ode(du, u, p, t)
    mum, delta = p
    y = u[1]
    du[1] = (mum - delta) * y
    return nothing
end

using Turing
@model function fit_ode(log_y_obs, times, prob)
    mum   ~ truncated(Normal(0.5, 0.3), 0.0, 1.0)
    delta ~ truncated(Normal(0.5, 0.3), 0.0, 1.0)
    N0    ~ LogNormal(log(1_630_000), 0.1)
    sigma ~ truncated(Normal(0, 1), 0, Inf)   
    
    pr = remake(prob;
                p = [mum, delta],
                u0 = [N0],
                tspan = (times[1], times[end]))
    
    # solve exactly at data times
    sol = solve(pr, Tsit5();  
                abstol = 1e-6, reltol = 1e-6,
                saveat = times)

    # likelihood at observation times
    log_y_pred = log.(Array(sol)[1, :] .+ 1e-9)
    log_y_obs ~ arraydist(Normal.(log_y_pred, sigma))
end

using DifferentialEquations

u0 = [y_obs[1]]
p = [0.5, 0.5]
tspan = (times[1], times[end])
prob = ODEProblem(ode, u0, tspan, p)

model    = fit_ode(log_y_obs, times, prob)
chain    = sample(model, NUTS(1000, .95), MCMCSerial(), 1000, 4; progress=true)

priors = Dict{Symbol,Distribution}(
    :mum   => truncated(Normal(0.5, 0.3), 0.0, 1.0),
    :delta => truncated(Normal(0.5, 0.3), 0.0, 1.0),
    :N0    => LogNormal(log(1_630_000.0), 0.1),
    :sigma => truncated(Normal(0, 1.0), 0.0, Inf)
)

order = [:mum, :delta, :N0, :sigma]
p1 = plot_trace_with_priors(chain; priors=priors, var_order=order, per_chain_density=true)  # also per-chain densities
savefig(p1, "../figures/growth_death_trace.png")

init_syms = [:N0]
param_syms = [:mum, :delta]
y_obs = y_obs
t_obs = times

p2 = overlay_posterior_on_observed(
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
savefig(p2, "../figures/growth_death_posterior.png")