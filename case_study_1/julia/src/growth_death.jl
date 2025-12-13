## Cell 5 ##

using CSV, DataFrames

df = CSV.read("../../../case_study_1/python/data/phaeocystis_control.csv", DataFrame)

times    = df.times
y_obs    = df.cells
log_y_obs = log.(y_obs .+ 1e-9)

