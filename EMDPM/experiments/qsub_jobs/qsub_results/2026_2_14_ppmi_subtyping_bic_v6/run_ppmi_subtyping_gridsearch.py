import numpy as np
import pandas as pd
import os
from EMDPM.subtyping_em_transformer import SubtypingEM
from EMDPM.utils import initialize_f_eigen, fit_mixedlm_beta_from_clinical, solve_system
from EMDPM.optimizer_beta import beta_loss
from sklearn.model_selection import train_test_split, KFold
from itertools import product

N_FOLDS = 3
CV_RANDOM_STATE = 75

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--candidate", type=int, default=0)
args = parser.parse_args()
current_candidate = args.candidate

df = pd.read_csv("/home/dsemchin/data/data_ppmi_pd.csv")
df_K = pd.read_csv("/home/dsemchin/data/mean_NORM_con_22.csv")
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

# Best practice: n_subtypes (K) is model order — select by BIC. Other params tuned by CV per K.
# Grid is per-K: for each K we run a full grid over hyperparameters; then choose K by min BIC(best_CV_per_K).
# Post-hoc: for each K in n_subtypes_list, take the run with best cv_mean_lse; then choose K with lowest bic.
N_SUBTYPES_LIST = [2, 3, 4]  # candidate number of subtypes; select among these by BIC

# Hyperparameter grid (no n_subtypes): tuned by CV within each K
param_grid_hyper = {
    "lambda_f": [1.0, 1.2, 1.4],
    "lambda_cog": [0.0, 0.05, 0.1],
    "lambda_scalar": [0.6, 0.8, 0.95],
    "lambda_jsd": [0, 10, 50, 100, 1000],
    "lambda_beta": [0.0]
}
hyper_names = list(param_grid_hyper.keys())
hyper_values = list(param_grid_hyper.values())
hyper_combinations = list(product(*hyper_values))
n_hyper_per_K = len(hyper_combinations)
total_combinations = len(N_SUBTYPES_LIST) * n_hyper_per_K

print(f"\n=== Grid Search Setup (two-level: K by BIC, hyperparams by CV) ===")
print(f"n_subtypes candidates: {N_SUBTYPES_LIST}")
print(f"Hyperparameter combinations per K: {n_hyper_per_K}")
print(f"Total combinations: {total_combinations}")
print(f"Current candidate: {current_candidate}")

if current_candidate >= total_combinations:
    import sys
    sys.exit(0)

# Map candidate index to (K, hyperparameter combo)
k_idx = current_candidate // n_hyper_per_K
sub_cand = current_candidate % n_hyper_per_K
n_subtypes = N_SUBTYPES_LIST[k_idx]
params = {name: hyper_combinations[sub_cand][i] for i, name in enumerate(hyper_names)}
params["n_subtypes"] = n_subtypes

print(f"\n=== Parameters for Candidate {current_candidate} (K={n_subtypes}, hyper idx={sub_cand}) ===")
for name, value in params.items():
    print(f"  {name}: {value}")

### THE MODEL

def build_estimator(params, K, f_init, t_max, rng_seed=75):
    return SubtypingEM(
        K=K,
        initial_f=f_init,
        n_subtypes=params["n_subtypes"],
        jac_toggle=True,
        max_iter=200,
        t_max=t_max,
        step=0.01,
        epsilon=1e-2,
        lambda_f=params["lambda_f"],
        lambda_cog=params["lambda_cog"],
        lambda_scalar=params["lambda_scalar"],
        lambda_jsd=params["lambda_jsd"],
        lambda_beta=params["lambda_beta"],
        verbose=1,
        rng=np.random.default_rng(rng_seed),
    )

# K-fold CV on X_train (by patient)
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
fold_lses = []
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(X_train)))):
    X_train_fold = [X_train[i] for i in train_idx]
    X_val_fold = [X_train[i] for i in val_idx]
    em_fold = build_estimator(params, K, f_init, t_max, rng_seed=75 + current_candidate * 100 + fold_idx)
    em_fold.fit(X_train_fold)
    tr = em_fold.transform(X_val_fold, use_cognitive_prior=True)
    lse_fold = em_fold._compute_val_score(X_val_fold, tr["beta"])
    fold_lses.append(lse_fold)
    if em_fold.verbose >= 1:
        print(f"  Fold {fold_idx + 1}/{N_FOLDS} val LSE: {lse_fold:.6f}")

cv_mean_lse = float(np.mean(fold_lses))
cv_std_lse = float(np.std(fold_lses))
print(f"CV mean LSE: {cv_mean_lse:.6f} ± {cv_std_lse:.6f} ({N_FOLDS} folds)")

# Refit on full X_train for final model, BIC, and validation metrics
subtyping_em = build_estimator(params, K, f_init, t_max, rng_seed=75)
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
param_str = "_".join([
    f"{name}{val:.3f}".replace(".", "p") if isinstance(val, float) 
    else f"{name}{val}" 
    for name, val in params.items()
])
out_path = os.path.join(out_dir, f"PPMI_subtyping_grid_betajsd_{current_candidate:03d}_{param_str}.npz")
bic_value = subtyping_em.bic_
lse_final = float(subtyping_em.lse_final)
n_obs = int(subtyping_em.n_obs_)
bic_k = int(subtyping_em._bic_n_params())
# BIC = neg2_log_L + k*ln(n); save both terms for transparency and reproducibility
bic_penalty = bic_k * np.log(n_obs)
bic_neg2_log_L = bic_value - bic_penalty

print("Saved:", out_path)
print(f"CV mean LSE ({N_FOLDS}-fold): {cv_mean_lse:.6f} ± {cv_std_lse:.6f}")
print(f"Final Training LSE: {lse_final:.6f}")
print(f"BIC (lower is better): {bic_value:.4f}  [ -2*ln(L)={bic_neg2_log_L:.2f}, penalty k*ln(n)={bic_penalty:.2f}, k={bic_k}, n_obs={n_obs} ]")
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
         n_subtypes=params["n_subtypes"],
         n_subtypes_list=np.array(N_SUBTYPES_LIST),
         lambda_f=params["lambda_f"],
         lambda_cog=params["lambda_cog"],
         lambda_scalar=params["lambda_scalar"],
         lambda_jsd=params["lambda_jsd"],
         lambda_beta=params["lambda_beta"],
         param_grid_size=total_combinations,
         n_hyper_per_K=n_hyper_per_K,
         k_idx=k_idx,
         sub_cand=sub_cand,
         # BIC (for model order selection: choose K by min BIC among best-CV run per K)
         bic=bic_value,
         bic_neg2_log_L=bic_neg2_log_L,
         bic_penalty=bic_penalty,
         n_obs=n_obs,
         bic_n_params=bic_k,
         lse_final=lse_final,
         # K-fold CV on training set (for hyperparameter selection within each K)
         cv_mean_lse=cv_mean_lse,
         cv_std_lse=cv_std_lse,
         cv_variance=float(cv_std_lse**2),
         cv_per_fold_lse=np.array(fold_lses),
         n_folds=N_FOLDS,
         # For BIC recompute without reloading data
         sse_per_biomarker=np.array(subtyping_em._sse_per_biomarker),
         n_obs_rows=int(subtyping_em._n_obs_rows))
print("Saved:", out_path)
print(f"CV mean LSE: {cv_mean_lse:.6f}")
print(f"Final LSE: {subtyping_em.lse_history[-1]:.6f}")
print(f"BIC: {bic_value:.4f}")

