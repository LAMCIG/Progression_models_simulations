import numpy as np
import pandas as pd
import os
from EMDPM.subtyping_em_transformer import SubtypingEM
from EMDPM.utils import initialize_f_eigen, fit_mixedlm_beta_from_clinical, solve_system
from EMDPM.optimizer_beta import beta_loss
from sklearn.model_selection import train_test_split
from itertools import product

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--candidate", type=int, default=0)
args = parser.parse_args()
current_candidate = args.candidate

df = pd.read_csv("/data01/bgutman/MRI_data/PPMI/data_ppmi_pd.csv")
df_K = pd.read_csv("/data01/bgutman/LEGACY/Skoltech/datasets/Connectomes/mean_NORM_con_22.csv")
#df_K = pd.read_csv("/data01/bgutman/LEGACY/Skoltech/datasets/Connectomes/mean_NORM_con_min_edges=3__sparsity=0.3.csv")
n_biomarkers = 68

## remove non-longitudinal observations
print("original size:", df.shape)
relevant_cols = [col for col in df.columns if col.startswith(('L_', 'R_')) and ('_thickavg' in col or '_thickavg_resid' in col)]
relevant_cols += ["MCATOT", "TD_score", "PIGD_score"]
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=relevant_cols)

print("after drop na", df.shape)
subj_counts = df['subj_id'].value_counts()
num_unique = (subj_counts == 1).sum()
print("one time subj_id:", num_unique)

longitudinal_ids = subj_counts[subj_counts > 1].index
df = df[df['subj_id'].isin(longitudinal_ids)].copy()
df = df.drop_duplicates(subset=["subj_id", "time"])
print("after drop dupes", df.shape)

X_obs = df[[col for col in df.columns if (col.startswith(('L_', 'R_')) and col.endswith('_thickavg') and not col.endswith('_thickavg_resid'))]]

biomarker_names = [col for col in df.columns 
                   if col.startswith(('L_', 'R_')) and 
                   col.endswith('_thickavg') and 
                   not col.endswith('_thickavg_resid')]

print(biomarker_names)

# X_obs = df[small_region_set]
X_obs = X_obs.to_numpy()
X_obs = np.max(X_obs, axis=0) - X_obs

print("nans in X:", np.isnan(X_obs).sum())
print("infs in X:", np.isinf(X_obs).sum())

## connectivity matrix to numpy
K = df_K.drop(df_K.columns[0], axis=1).to_numpy()
np.fill_diagonal(K, 0)
print(K.shape, type(K))

# normalization
row_sums = K.sum(axis=1)
median_row_sum = np.median(row_sums)
K = K / median_row_sum

t_max = 40
print("X.size: ", X_obs.shape)

ids = df["subj_id"].to_numpy()
dt = df["time"].to_numpy()/12 # convert to years
#cog = df["MCATOT"].values#,"TD_score","PIGD_score"]].values
cog = df[["MCATOT","TD_score","PIGD_score"]].to_numpy()
nhy = df["NHY"].to_numpy()
print("nans in cog:", np.isnan(cog).sum())
print("infs in cog:", np.isinf(cog).sum())

df["NSD_STAGE"] = df["NSD_STAGE"].replace({"Not NSD": 0, "2b": 2})
df["NSD_STAGE"] = pd.to_numeric(df["NSD_STAGE"], errors='coerce')  # handles any remaining invalid entries

patient_df = df.copy()
patient_df["NHY"] = nhy  # add NHY array to df
grouped = patient_df.groupby("subj_id")[["NSD_STAGE", "NHY"]].mean().dropna()

nsd = df["NSD_STAGE"].to_numpy()

### INITIALIZE BETA FROM CLINICAL SCORES
initial_beta, pid_to_beta, _ = fit_mixedlm_beta_from_clinical(
    df=df, 
    ids=ids,
    dt=dt, 
    t_max=t_max, 
    verbose=True,
    rng=np.random.default_rng(75)
)

### INITIALIZE F - Use a single fixed f_init (candidate 0)
f_init_list = initialize_f_eigen(K=K, 
                                 jitter_strength=0.05, 
                                 n_eigs=100,
                                 rng=np.random.RandomState(75)
                                 )
f_init = f_init_list[0]  # Fixed to candidate 0

# Ensure f_init is 1D (flatten if needed)
if isinstance(f_init, list):
    f_init = f_init[0]
f_init = np.ravel(f_init)

### DATA SPLIT

def create_patient_list(X_obs, ids, dt, cog, initial_beta=None):
    unique_ids = np.unique(ids)
    id_to_index = {pid: idx for idx, pid in enumerate(unique_ids)}

    patient_list = []
    for pid in unique_ids:
        mask = (ids == pid)
        patient_data = {
            "id": pid,
            "X_obs": X_obs[mask],
            "dt": dt[mask],
            "cog": cog[mask],
            "nhy": nhy[mask]
        }
        if initial_beta is not None:
            patient_data["initial_beta"] = initial_beta[id_to_index[pid]]
        patient_list.append(patient_data)

    return patient_list

X = create_patient_list(X_obs, ids, dt, cog, initial_beta)
X_train, X_val = train_test_split(X, test_size=0.2, random_state=75)

# Define parameter grid
param_grid = {
    "lambda_f": [0.4, 0.5, 0.6, 0.7],
    "lambda_cog": [0.0, 0.1, 0.25],
    "lambda_scalar": [1.1],
    "lambda_jsd": [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
    "lambda_beta": [0.0]
}


# Create all combinations
param_names = list(param_grid.keys())
param_values = list(param_grid.values())
param_combinations = list(product(*param_values))

total_combinations = len(param_combinations)
print(f"\n=== Grid Search Setup ===")
print(f"Total parameter combinations: {total_combinations}")
print(f"Current candidate: {current_candidate}")

if current_candidate >= total_combinations:
    # Exit silently for out-of-range candidates (PBS array may be larger than needed)
    import sys
    sys.exit(0)  # Exit gracefully so PBS doesn't treat it as a failure

# Get parameters for this candidate
params = {name: param_combinations[current_candidate][i] 
          for i, name in enumerate(param_names)}

print(f"\n=== Parameters for Candidate {current_candidate} ===")
for name, value in params.items():
    print(f"  {name}: {value}")

### THE MODEL

n_subtypes = 2

subtyping_em = SubtypingEM(
    K=K,
    initial_f=f_init,
    n_subtypes=n_subtypes,
    jac_toggle=True,
    max_iter=100,
    t_max=t_max,
    step=0.01,
    epsilon=1e-2,
    lambda_f=params["lambda_f"],
    lambda_cog=params["lambda_cog"],
    lambda_scalar=params["lambda_scalar"],
    lambda_jsd=params["lambda_jsd"],
    lambda_beta=params["lambda_beta"],
    verbose=1,
    rng=np.random.default_rng(75)
)

subtyping_em.fit(X_train)
# Transform on validation with cognitive priors (default) and without priors
transform_results_with_cog = subtyping_em.transform(X_val, use_cognitive_prior=True)
transform_results_no_cog = subtyping_em.transform(X_val, use_cognitive_prior=False)

# Extract beta and subtype assignments
beta_val_with_cog = transform_results_with_cog['beta']
val_assignments_with_cog = transform_results_with_cog['subtype']

beta_val_no_cog = transform_results_no_cog['beta']
val_assignments_no_cog = transform_results_no_cog['subtype']

# Compute validation LSE for both settings
val_lse_with_cog = subtyping_em._compute_val_score(X_val, beta_val_with_cog)
val_lse_no_cog = subtyping_em._compute_val_score(X_val, beta_val_no_cog)

# Save results in self-contained experiment folder
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "results")
os.makedirs(out_dir, exist_ok=True)

# Create filename with parameter values for easy identification
param_str = "_".join([f"{name}{val:.3f}".replace(".", "p") for name, val in params.items()])
out_path = os.path.join(out_dir, f"PPMI_subtyping_grid_betajsd_{current_candidate:03d}_{param_str}.npz")
print("Saved:", out_path)
print(f"Final Training LSE: {subtyping_em.lse_history[-1]:.6f}")
print(f"Validation LSE (with cog): {val_lse_with_cog:.6f}")
print(f"Validation LSE (no cog): {val_lse_no_cog:.6f}")


# Get unique patient IDs for train and val sets
train_ids = [p["id"] for p in X_train]
val_ids_unique = [p["id"] for p in X_val]

# Create assignment arrays aligned with patient IDs
all_train_ids = np.array(train_ids)
all_val_ids = np.array(val_ids_unique)
train_assignments_array = subtyping_em.final_assignments
# val_assignments_array already extracted from transform_results above

np.savez(out_path,
         theta_history=np.array(subtyping_em.theta_history),
         cog_history=np.array(subtyping_em.cog_regression_history),  # 3D: (n_subtypes, n_cog_features+1, max_iter)
         beta_history=np.array(subtyping_em.beta_history),
         lse_history=np.array(subtyping_em.lse_history),
         assignment_history=np.array(subtyping_em.assignment_history),
         beta_val=np.array(beta_val_with_cog),  # backward-compatible: with cognitive priors
         beta_val_with_cog=np.array(beta_val_with_cog),
         beta_val_no_cog=np.array(beta_val_no_cog),
         val_assignments_with_cog=np.array(val_assignments_with_cog),
         val_assignments_no_cog=np.array(val_assignments_no_cog),
         val_lse_with_cog=val_lse_with_cog,
         val_lse_no_cog=val_lse_no_cog,
         candidate=current_candidate,
         f_init=f_init,
         # Subtype assignments
         train_assignments=train_assignments_array,
         val_assignments=val_assignments_with_cog,  # backward-compatible: with cognitive priors
         train_ids=all_train_ids,
         val_ids=all_val_ids,
         final_assignments=subtyping_em.final_assignments,
         cluster_f=np.array(subtyping_em.cluster_f),
         cluster_cog_a=np.array(subtyping_em.cluster_cog_a),  # (n_subtypes, n_cog_features)
         cluster_cog_b=np.array(subtyping_em.cluster_cog_b),  # (n_subtypes,)
         final_scalar_K=subtyping_em.final_scalar_K,
         final_s=subtyping_em.final_s,
         # Grid search parameters
         lambda_f=params["lambda_f"],
         lambda_cog=params["lambda_cog"],
         lambda_scalar=params["lambda_scalar"],
         lambda_jsd=params["lambda_jsd"],
         lambda_beta=params["lambda_beta"],
         param_grid_size=total_combinations)
print("Saved:", out_path)
print(f"Final LSE: {subtyping_em.lse_history[-1]:.6f}")

