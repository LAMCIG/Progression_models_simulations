import os
import sys
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from EMDPM.model_generator import generate_logistic_model
from EMDPM.synthetic_data_generator import generate_synthetic_data
from EMDPM.em_transformer import EM

# ----------------------
# PARAMETER SWEEP SETUP
# ----------------------
noise_levels = [0.0, 0.1, 0.2, 0.3]
n_biomarkers_list = [5, 10, 20, 40]
jacobian_options = [False, True]

# Fixed params
t_max = 12
n_patients = 200
n_patient_obs = 3
num_iterations = 50
step = 0.01

# PBS_ARRAYID comes in as a string
array_id = int(os.environ.get("PBS_ARRAYID", 1)) - 1

# Make all combinations
grid = []
for noise in noise_levels:
    for n_biomarkers in n_biomarkers_list:
        for use_jacobian in jacobian_options:
            grid.append((noise, n_biomarkers, use_jacobian))

# Select config for this job
noise_level, n_biomarkers, use_jacobian = grid[array_id]

# Create timestamped output folder
output_dir = f"results/noise_{noise_level}_biomarkers_{n_biomarkers}_jac_{use_jacobian}"
os.makedirs(output_dir, exist_ok=True)

# -------------------
# RUN EM EXPERIMENT
# -------------------

# Simulate model and data
t, x_true, K = generate_logistic_model(n_biomarkers=n_biomarkers, t_max=t_max)
df, beta_true_dict = generate_synthetic_data(
    n_biomarkers=n_biomarkers,
    t_max=t_max,
    noise_level=noise_level,
    n_patients=n_patients,
    n_patient_obs=n_patient_obs,
    x_true=x_true,
    t=t
)

# Fit EM
em_model = EM(K=K, num_iterations=num_iterations, t_max=t_max, step=step, use_jacobian=use_jacobian)
em_model.fit(df)

# Save results
with open(os.path.join(output_dir, "theta_iter.pkl"), "wb") as f:
    pickle.dump(em_model.theta_iter_, f)

with open(os.path.join(output_dir, "beta_iter.pkl"), "wb") as f:
    pickle.dump(em_model.beta_iter_, f)

with open(os.path.join(output_dir, "lse_array.pkl"), "wb") as f:
    pickle.dump(em_model.lse_array_, f)

# Save metadata
with open(os.path.join(output_dir, "config.txt"), "w") as f:
    f.write(f"Noise level: {noise_level}\n")
    f.write(f"Biomarkers: {n_biomarkers}\n")
    f.write(f"Use Jacobian: {use_jacobian}\n")
    f.write(f"Date: {datetime.now()}\n")

print(f"Completed experiment: noise={noise_level}, biomarkers={n_biomarkers}, jacobian={use_jacobian}")
