import numpy as np
import pandas as pd
import os
from EMDPM.subtyping_em_transformer import SubtypingEM, run_multiple_initializations_parallel
from EMDPM.utils import initialize_f_eigen, fit_mixedlm_beta_from_clinical, solve_system
from EMDPM.optimizer_beta import beta_loss
from sklearn.model_selection import train_test_split

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--n_init", type=int, default=100, help="Number of initializations")
parser.add_argument("--run_index", type=int, default=0, help="Run index for this job")
args = parser.parse_args()

n_initializations = args.n_init
run_index = args.run_index

df = pd.read_csv("/data01/bgutman/MRI_data/PPMI/data_ppmi_pd.csv")
df_K = pd.read_csv("/data01/bgutman/LEGACY/Skoltech/datasets/Connectomes/mean_NORM_con_22.csv")
n_biomarkers = 68

## remove non-longitudinal observations
print("original size:", df.shape)
relevant_cols = [col for col in df.columns if col.startswith(('L_', 'R_')) and ('_thickavg' in col or '_thickavg_resid' in col)]
relevant_cols += ["MCATOT", "TD_score", "PIGD_score"]
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=relevant_cols)

print("after drop na", df.shape)
subj_counts = df['subj_id'].value_counts()
longitudinal_ids = subj_counts[subj_counts > 1].index
df = df[df['subj_id'].isin(longitudinal_ids)].copy()
df = df.dropna(subset=relevant_cols)
df = df.drop_duplicates(subset=["subj_id", "time"])
print("after drop dupes", df.shape)

X_obs = df[[col for col in df.columns if (col.startswith(('L_', 'R_')) and col.endswith('_thickavg') and not col.endswith('_thickavg_resid'))]]

biomarker_names = [col for col in df.columns 
                   if col.startswith(('L_', 'R_')) and 
                   col.endswith('_thickavg') and 
                   not col.endswith('_thickavg_resid')]

X_obs = X_obs.to_numpy()
X_obs = np.max(X_obs, axis=0) - X_obs

print("nans in X:", np.isnan(X_obs).sum())
print("infs in X:", np.isinf(X_obs).sum())

## connectivity matrix to numpy
K = df_K.drop(df_K.columns[0], axis=1).to_numpy()
np.fill_diagonal(K, 0)

# normalization
row_sums = K.sum(axis=1)
median_row_sum = np.median(row_sums)
K = K / median_row_sum

t_max = 40
print("X.size: ", X_obs.shape)

ids = df["subj_id"].to_numpy()
dt = df["time"].to_numpy()/12 # convert to years
cog = df[["MCATOT","TD_score","PIGD_score"]].to_numpy()
nhy = df["NHY"].to_numpy()
print("nans in cog:", np.isnan(cog).sum())
print("infs in cog:", np.isinf(cog).sum())

df["NSD_STAGE"] = df["NSD_STAGE"].replace({"Not NSD": 0, "2b": 2})
df["NSD_STAGE"] = pd.to_numeric(df["NSD_STAGE"], errors='coerce')

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

#   lambda_f: 0.6
#   lambda_cog: 0.2
#   lambda_scalar: 1.1
#   lambda_jsd: 0.4
#   lambda_beta: 0.01


### BEST PARAMETERS
best_params = {
    "lambda_f": 0.6,
    "lambda_cog": 0.2,
    "lambda_scalar": 1.1,
    "lambda_jsd": 0.4,
    "lambda_beta": 0.01
}

print(f"\n=== Multiple Initializations Setup ===")
print(f"Number of initializations: {n_initializations}")
print(f"Best parameters:")
for name, value in best_params.items():
    print(f"  {name}: {value}")

### THE MODEL - Run multiple initializations
n_subtypes = 2

em_kwargs = {
    'K': K,
    'initial_f': f_init,
    'n_subtypes': n_subtypes,
    'jac_toggle': True,
    'max_iter': 1000,  # Higher iteration count
    't_max': t_max,
    'step': 0.01,
    'epsilon': 1e-2, 
    'lambda_f': best_params["lambda_f"],
    'lambda_cog': best_params["lambda_cog"],
    'lambda_scalar': best_params["lambda_scalar"],
    'lambda_jsd': best_params["lambda_jsd"],
    'lambda_beta': best_params["lambda_beta"],
    'verbose': 1
}

# Run multiple initializations in parallel
print(f"\nRunning {n_initializations} initializations...")
results, best_idx = run_multiple_initializations_parallel(
    X=X_train,
    n_initializations=n_initializations,
    em_kwargs=em_kwargs,
    n_jobs=-1,
    prefer="processes",
    seed_offset=run_index * 1000  # Different seed offset per run
)

best_model = results[best_idx]['model']
best_final_lse = results[best_idx]['final_lse']
best_original_run_index = results[best_idx]['run_index']

print(f"\n=== Best Model Found ===")
print(f"Best initialization index (in successful runs): {best_idx}")
print(f"Best original run index: {best_original_run_index}")
print(f"Best final LSE: {best_final_lse:.6f}")

# Transform validation set with best model
beta_val = best_model.transform(X_val)

# Compute assignments for validation set using the best fitted model
val_X_obs_list = []
val_dt_list = []
val_ids_list = []
val_cog_list = []

for i, patient in enumerate(X_val):
    n = len(patient["dt"])
    val_X_obs_list.append(patient["X_obs"])
    val_dt_list.append(patient["dt"])
    val_ids_list.append(np.full(n, i))
    val_cog_list.append(patient["cog"] if patient["cog"].ndim == 2 else patient["cog"].reshape(-1, 1))

val_X_obs = np.vstack(val_X_obs_list)
val_dt = np.concatenate(val_dt_list)
val_ids = np.concatenate(val_ids_list)
val_cog = np.vstack(val_cog_list)

# Compute assignments for validation set using per-subtype cognitive parameters
val_assignments = []
val_unique_ids = np.unique(val_ids)

for idx, patient_id in enumerate(val_unique_ids):
    mask = (val_ids == patient_id)
    X_obs_i = val_X_obs[mask, :]
    dt_i = val_dt[mask]
    cog_i = val_cog[mask, :]
    beta_i = beta_val[idx]
    
    best_error = np.inf
    best_subtype = 0
    
    # Try each cluster and find the one with lowest reconstruction error
    for subtype in range(n_subtypes):
        f_cluster = np.ravel(best_model.cluster_f[subtype])
        theta_cluster = np.concatenate([f_cluster, best_model.final_s, [best_model.final_scalar_K]])
        X_pred_cluster = solve_system(np.zeros(X_obs_i.shape[1]), f_cluster, K, best_model.t_span, best_model.final_scalar_K)
        
        # Use subtype-specific cognitive parameters
        cog_a_subtype = best_model.cluster_cog_a[subtype]
        cog_b_subtype = best_model.cluster_cog_b[subtype]
        
        # Compute reconstruction error
        error = beta_loss(
            beta_i, X_obs_i, dt_i, X_pred_cluster, best_model.t_span,
            cog_i, cog_a_subtype, cog_b_subtype, theta_cluster, best_model.lambda_cog
        )
        
        if error < best_error:
            best_error = error
            best_subtype = subtype
    
    val_assignments.append(best_subtype)

# Create output directory
out_dir = "/home/dsemchin/Progression_models_simulations/EMDPM/experiments/qsub_jobs/qsub_results/ppmi_multinit_bestparams"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, f"PPMI_subtyping_multinit_best.npz")

# Get unique patient IDs for train and val sets
train_ids = [p["id"] for p in X_train]
val_ids_unique = [p["id"] for p in X_val]

# Create assignment arrays aligned with patient IDs
all_train_ids = np.array(train_ids)
all_val_ids = np.array(val_ids_unique)
train_assignments_array = best_model.final_assignments
val_assignments_array = np.array(val_assignments)

# Save best model results
np.savez(out_path,
         theta_history=np.array(best_model.theta_history),
         cog_history=np.array(best_model.cog_regression_history),  # 3D: (n_subtypes, n_cog_features+1, max_iter)
         beta_history=np.array(best_model.beta_history),
         lse_history=np.array(best_model.lse_history),
         assignment_history=np.array(best_model.assignment_history),
         beta_val=np.array(beta_val),
         f_init=f_init,
         # Subtype assignments
         train_assignments=train_assignments_array,
         val_assignments=val_assignments_array,
         train_ids=all_train_ids,
         val_ids=all_val_ids,
         final_assignments=best_model.final_assignments,
         cluster_f=np.array(best_model.cluster_f),
         cluster_cog_a=np.array(best_model.cluster_cog_a),  # (n_subtypes, n_cog_features)
         cluster_cog_b=np.array(best_model.cluster_cog_b),  # (n_subtypes,)
         final_scalar_K=best_model.final_scalar_K,
         final_s=best_model.final_s,
         # Best parameters
         lambda_f=best_params["lambda_f"],
         lambda_cog=best_params["lambda_cog"],
         lambda_scalar=best_params["lambda_scalar"],
         lambda_jsd=best_params["lambda_jsd"],
         lambda_beta=best_params["lambda_beta"],
         n_initializations=n_initializations,
         best_init_index=best_original_run_index,
         n_successful=len(results))

print("Saved:", out_path)
print(f"Final LSE: {best_final_lse:.6f}")

