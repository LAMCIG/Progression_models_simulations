import numpy as np
import pandas as pd
import os
from EMDPM.subtyping_em_transformer import SubtypingEM
from EMDPM.utils import initialize_f_eigen, fit_mixedlm_beta_from_clinical, solve_system
from EMDPM.optimizer_beta import beta_loss
from sklearn.model_selection import train_test_split

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

### INITIALIZE F
f_init_list = initialize_f_eigen(K=K, 
                                 jitter_strength=0.05, 
                                 n_eigs=100,
                                 rng=np.random.RandomState(75)
                                 )
assert 0 <= current_candidate < f_init_list.shape[0]
f_init = f_init_list[current_candidate]

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

### THE MODEL

n_subtypes = 2

subtyping_em = SubtypingEM(
    K=K,
    initial_f=f_init,
    n_subtypes=n_subtypes,
    jac_toggle=True,
    max_iter=30,
    t_max=t_max,
    step=0.01,
    epsilon=1e-2,
    lambda_f=1.0,
    lambda_cog=0.01,
    lambda_scalar=0.3,
    lambda_jsd=0.1,  # JSD regularization for beta separation
    lambda_beta=0.15,  # L2 regularization on beta values
    verbose=1,
    rng=np.random.default_rng(75)
)

subtyping_em.fit(X_train)
beta_val = subtyping_em.transform(X_val)

# Compute assignments for validation set using the fitted model
# Prepare validation data in the same format as training
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

# Compute assignments for validation set
val_assignments = []
val_unique_ids = np.unique(val_ids)
val_cog_a = subtyping_em.cog_regression_history[:-1, -1]
val_cog_b = subtyping_em.cog_regression_history[-1, -1]

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
        f_cluster = np.ravel(subtyping_em.cluster_f[subtype])
        theta_cluster = np.concatenate([f_cluster, subtyping_em.final_s, [subtyping_em.final_scalar_K]])
        X_pred_cluster = solve_system(np.zeros(X_obs_i.shape[1]), f_cluster, K, subtyping_em.t_span, subtyping_em.final_scalar_K)
        
        # Compute reconstruction error
        error = beta_loss(
            beta_i, X_obs_i, dt_i, X_pred_cluster, subtyping_em.t_span,
            cog_i, val_cog_a, val_cog_b, theta_cluster, subtyping_em.lambda_cog
        )
        
        if error < best_error:
            best_error = error
            best_subtype = subtype
    
    val_assignments.append(best_subtype)

out_dir = "/home/dsemchin/Progression_models_simulations/EMDPM/experiments/qsub_jobs/qsub_results"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"PPMI_subtyping_{current_candidate}.npz")

# Get unique patient IDs for train and val sets
train_ids = [p["id"] for p in X_train]
val_ids_unique = [p["id"] for p in X_val]

# Create assignment arrays aligned with patient IDs
all_train_ids = np.array(train_ids)
all_val_ids = np.array(val_ids_unique)
train_assignments_array = subtyping_em.final_assignments
val_assignments_array = np.array(val_assignments)

np.savez(out_path,
         theta_history=np.array(subtyping_em.theta_history),
         cog_history=np.array(subtyping_em.cog_regression_history),
         beta_history=np.array(subtyping_em.beta_history),
         lse_history=np.array(subtyping_em.lse_history),
         assignment_history=np.array(subtyping_em.assignment_history),
         beta_val=np.array(beta_val),
         candidate=current_candidate,
         f_init=f_init,
         # Subtype assignments
         train_assignments=train_assignments_array,
         val_assignments=val_assignments_array,
         train_ids=all_train_ids,
         val_ids=all_val_ids,
         final_assignments=subtyping_em.final_assignments,
         cluster_f=np.array(subtyping_em.cluster_f),
         final_scalar_K=subtyping_em.final_scalar_K,
         final_s=subtyping_em.final_s)
print("Saved:", out_path)

