## Cell 5 ##

using CSV, DataFrames

cells = CSV.read("../../../case_study_2/python/data/total_cells.csv", DataFrame)
death = CSV.read("../../../case_study_2/python/data/death_percentage.csv", DataFrame)

cells_times  = cells[end-14:end, :1]
cells_obs    = cells[end-14:end, :2] * 1e6
log_cells_obs = log.(cells_obs .+ 1e-9)

death_times = death[end-14:end, :1]
death_obs   = death[end-14:end, :2] .* (cells_obs ./ 100)
log_death_obs = log.(death_obs .+ 1e-9)

## Cell 6 ##

function ode(du, u, p, t)
    P, D = u
    r, K, delta = p

    du[1] = r * (1 - P / K) * P - delta * P
    du[2] = delta * P

    return nothing
end

## Cell 7 ##

using Turing

@model function fit_ode(log_cells_obs, log_death_obs, times, prob)
    r ~ Uniform(0.5, 1.0)
    K ~ Uniform(1e6, 4e7)
    delta ~ Uniform(0.0, 0.15)

    P0 ~ Uniform(1e5, 3e5)
    D0 ~ Uniform(1e4, 7e4)

    sigma_live ~ truncated(Normal(0, 3), 0, Inf)
    sigma_dead ~ truncated(Normal(0, 3), 0, Inf)

    pr = remake(prob;
                p = [r, K, delta],
                u0 = [P0, D0],
                tspan = (times[1], times[end]))
    
    # solve exactly at data times
    sol = solve(pr, Tsit5();
                abstol = 1e-6, reltol = 1e-6,
                saveat = times)
    S = Array(sol) 

    log_cells_pred = log.((S[1, :] + S[2, :]) .+ 1e-9)
    log_death_pred = log.(S[2, :] .+ 1e-9)

    log_cells_obs ~ arraydist(Normal.(log_cells_pred, sigma_live))
    log_death_obs ~ arraydist(Normal.(log_death_pred, sigma_dead))

end

## Cell 8 ##

using DifferentialEquations

u0 = [log_cells_obs[1], log_death_obs[1]]
p = [0.5, 1e6, 0.0] 
tspan = (cells_times[1], cells_times[end])
prob = ODEProblem(ode, u0, tspan, p)

model    = fit_ode(log_cells_obs, log_death_obs, cells_times, prob)
chain    = sample(model, NUTS(1000, .95), MCMCSerial(), 1000, 4; progress=false)
 

priors = Dict{Symbol,Distribution}(
    :r     => Uniform(0.5, 1.0),
    :K     => Uniform(1e6, 4e7),
    :delta => Uniform(0.0, 0.15),
    :P0    => Uniform(1e5, 3e5),
    :D0    => Uniform(1e4, 7e4),
    :sigma_live => truncated(Normal(0, 3), 0, Inf),
    :sigma_dead => truncated(Normal(0, 3), 0, Inf)
)

order = [:r, :K, :delta, :P0, :D0, :sigma_live, :sigma_dead]

plot_trace_with_priors(chain; priors=priors, var_order=order, per_chain_density=true)  # also per-chain densities

init_syms = [:P0, :D0]
param_syms = [:r, :K, :delta]
t_obs = cells_times
y_obs = hcat(cells_obs, death_obs)

plt = overlay_posterior_on_observed(
    chain, ode, t_obs, y_obs;
    init_syms=init_syms,
    param_syms=param_syms,
    which_states=[1, 2],     # choose states to plot
    pred_transforms=[u -> u[1] + u[2], u -> u[2]], # column1=total(P+D), column2=dead(D)
    legend=:topleft,
    n_draws=150,            # how many posterior paths to overlay
    plot_ribbon=true,       # median ± CI band
    ribbon_q=(0.1, 0.9),    # CI limits
    logy=false
)
display(plt)

