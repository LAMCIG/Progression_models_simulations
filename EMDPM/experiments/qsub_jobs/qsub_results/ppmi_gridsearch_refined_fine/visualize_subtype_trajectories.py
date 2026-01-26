#!/usr/bin/env python3
"""
Visualize subtype trajectories on brain at a single time point.

To switch subtypes or time points, change SUBTYPE and TIME_POINT below and run again.
"""

import os
# Set Qt environment variables BEFORE importing any Qt-based libraries
# This allows the script to run on headless servers
# 
# If 'offscreen' doesn't work, try using Xvfb (virtual framebuffer):
#   1. Install: yum install xorg-x11-server-Xvfb  (or apt-get install xvfb)
#   2. Run: xvfb-run -a python visualize_subtype_trajectories.py
# 
# Or if you have X11 forwarding enabled via SSH:
#   - Comment out the line below and run normally
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Use offscreen rendering

import numpy as np
import pandas as pd
from EMDPM.utils import solve_system
from EMDPM.brain_utils import visualize_brain_region_statistics

# ============================================================================
# CONFIGURATION - Change these values and run again
# ============================================================================
SUBTYPE = 0  # 0 or 1
TIME_POINT = 10  # Time in years (0-40)
# ============================================================================

# Paths
result_dir = "/home/dsemchin/Progression_models_simulations/EMDPM/experiments/qsub_jobs/qsub_results/ppmi_gridsearch_refined_fine"
result_file = os.path.join(result_dir, "PPMI_subtyping_grid_103_lambda_f0p300_lambda_cog0p100_lambda_scalar0p500_lambda_jsd0p100_lambda_beta1p000.npz")
visualizations_dir = os.path.join(result_dir, "visualizations")
os.makedirs(visualizations_dir, exist_ok=True)

# Data paths
data_path = "/data01/bgutman/MRI_data/PPMI/data_ppmi_pd.csv"
connectome_path = "/data01/bgutman/LEGACY/Skoltech/datasets/Connectomes/mean_NORM_con_22.csv"

print(f"Loading candidate 103 results...")
print(f"Visualizing Subtype {SUBTYPE} at {TIME_POINT} years")
print(f"Results will be saved to: {visualizations_dir}\n")

# Load results
best_data = np.load(result_file, allow_pickle=True)

# Extract model parameters
cluster_f = best_data["cluster_f"]  # (n_subtypes, n_biomarkers)
final_scalar_K = best_data["final_scalar_K"]
final_s = best_data["final_s"]
n_biomarkers = cluster_f.shape[1]

print(f"Model parameters loaded:")
print(f"  Number of biomarkers: {n_biomarkers}")
print(f"  Final scalar_K: {final_scalar_K:.6f}")
print(f"  Final s range: [{np.min(final_s):.4f}, {np.max(final_s):.4f}]\n")

# Load connectivity matrix
df_K = pd.read_csv(connectome_path)
K = df_K.drop(df_K.columns[0], axis=1).to_numpy()
np.fill_diagonal(K, 0)

# Normalize connectivity matrix
row_sums = K.sum(axis=1)
median_row_sum = np.median(row_sums)
K = K / median_row_sum

# Load biomarker names
df = pd.read_csv(data_path)
biomarker_names = [col for col in df.columns 
                   if col.startswith(('L_', 'R_')) and 
                   col.endswith('_thickavg') and 
                   not col.endswith('_thickavg_resid')]

print(f"Biomarker names loaded: {len(biomarker_names)} regions\n")

# Time configuration
t_max = 40
t_span = np.linspace(0, t_max, int(t_max/0.01))

# Get forcing term for this subtype
f_subtype = cluster_f[SUBTYPE]
print(f"Subtype {SUBTYPE} forcing term range: [{np.min(f_subtype):.4f}, {np.max(f_subtype):.4f}]\n")

# Compute full trajectory
x0 = np.zeros(n_biomarkers)
Xtraj_full = solve_system(x0, f_subtype, K, t_span, final_scalar_K)
# Apply scaling
Xtraj_scaled = Xtraj_full * final_s[:, None]  # (n_biomarkers, n_timepoints)

print(f"Trajectory computed. Shape: {Xtraj_scaled.shape}")
print(f"Trajectory value range: [{np.min(Xtraj_scaled):.4f}, {np.max(Xtraj_scaled):.4f}]\n")

# Find closest time index
t_idx = np.argmin(np.abs(t_span - TIME_POINT))
actual_t = t_span[t_idx]

print(f"Visualizing time point: {TIME_POINT} years (actual: {actual_t:.2f} years)...")

# Extract trajectory values at this time point
traj_values = Xtraj_scaled[:, t_idx]  # (n_biomarkers,)

print(f"Trajectory value range at t={actual_t:.2f}: [{np.min(traj_values):.4f}, {np.max(traj_values):.4f}]\n")

# Create DataFrame for brain visualization
region_stats = pd.DataFrame({
    'P-value': [0.001] * n_biomarkers,  # Set all to significant so they all show
    'FX_size': traj_values
}, index=biomarker_names)

# Create visualization
visualize_brain_region_statistics(
    region_stats,
    colormap='coolwarm',
    cbar_string=f'Subtype {SUBTYPE} - Trajectory at {actual_t:.1f} years',
    p_val_threshold=0.05
)
