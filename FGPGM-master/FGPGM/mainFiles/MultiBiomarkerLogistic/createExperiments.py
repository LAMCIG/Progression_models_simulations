# -*- coding: utf-8 -*-
"""
Creates and saves all the files used later by FGPGM for parameter estimation.
Must be run from a folder where the output should be saved.
"""
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from FGPGM.Experiments.multiBiomarkerLogistic import generate_logistic_model

seed = np.random.randint(0, 2**31 - 1)
np.random.seed(seed)
np.savetxt("seed.csv", np.array([seed]))

n_biomarkers = 5
t, x_true, K = generate_logistic_model(n_biomarkers=n_biomarkers,
                                       step=0.1,
                                       n_steps=100,
                                       neg_frac=0.01,
                                       connectivity_matrix_type='offdiag')
# x_true shape: (n_biomarkers, len(t))

noise_level = 0.00
n_obs = 20
obs_indices = np.random.choice(len(t), size=n_obs, replace=False)
obs_indices.sort()

t_obs = t[obs_indices]
x_obs = x_true[:, obs_indices] + np.random.normal(0, noise_level, (n_biomarkers, n_obs))
x_obs = np.clip(x_obs, 0, 1)

# DEBUG
# plt.figure(figsize=(10, 4))
# for i in range(n_biomarkers):
#     plt.plot(t, x_true[i], label=f"true biomarker {i+1}", alpha=0.5)
#     plt.scatter(t_obs, x_obs[i], label=f"observed biomarker {i+1}", alpha=0.7)
# plt.legend()
# plt.savefig("biomarkers.png")
# plt.close()

# Save data in CSV files consistent with FGPGM pipeline
# FGPGM expects each row to be a time point and each column a state
# therefore transpose x_obs and x_true when saving.

np.savetxt("time.csv", t_obs)
np.savetxt("observations.csv", x_obs.T)
np.savetxt("trueStates.csv", x_true[:, obs_indices].T)

theta = np.array([0.01, 0.01])
np.savetxt("trueODEParams.csv", theta)

XInit = x_true[:, 0]
np.savetxt("XInit.csv", XInit)
print("Experiment data created successfully!")
