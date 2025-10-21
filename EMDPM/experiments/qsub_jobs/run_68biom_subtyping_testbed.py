# testing_bed_subtyping_from_npz.py
import numpy as np
import pandas as pd
import os
import copy

from EMDPM.subject_EM import SubjectEM
from EMDPM.subject_subtyping import cluster_and_plot
from EMDPM.utils import initialize_f_eigen

from sklearn.model_selection import train_test_split

# 1) load data
df = pd.read_csv("/data01/bgutman/MRI_data/PPMI/data_ppmi_pd.csv")
df_K = pd.read_csv("/data01/bgutman/LEGACY/Skoltech/datasets/Connectomes/mean_NORM_con_min_edges=3__sparsity=0.3.csv")
n_biomarkers = 68

relevant_cols = [c for c in df.columns if c.startswith(('L_', 'R_')) and ('_thickavg' in c or '_thickavg_resid' in c)]
relevant_cols += ["MCATOT", "TD_score", "PIGD_score"]
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=relevant_cols)

subj_counts = df['subj_id'].value_counts()
longitudinal_ids = subj_counts[subj_counts > 1].index
df = df[df['subj_id'].isin(longitudinal_ids)].copy()
df = df.drop_duplicates(subset=["subj_id", "time"])

X_obs_df = df[[c for c in df.columns if (c.startswith(('L_', 'R_')) and c.endswith('_thickavg') and not c.endswith('_thickavg_resid'))]]
biomarker_names = list(X_obs_df.columns)

X_obs = X_obs_df.to_numpy()
X_obs = np.max(X_obs, axis=0) - X_obs

K = df_K.drop(df_K.columns[0], axis=1).to_numpy()
np.fill_diagonal(K, 0)
K = K / np.median(K.sum(axis=1))

ids = df["subj_id"].to_numpy()
dt = (df["time"].to_numpy() / 12.0).astype(float)
cog = df[["MCATOT","TD_score","PIGD_score"]].to_numpy()
nhy = df["NHY"].to_numpy()

# helper to build patient list
def create_patient_list(X_obs, ids, dt, cog, nhy):
    unique_ids = np.unique(ids)
    patient_list = []
    for pid in unique_ids:
        m = (ids == pid)
        patient_list.append({
            "id": pid,
            "X_obs": X_obs[m],
            "dt": dt[m],
            "cog": cog[m],
            "nhy": nhy[m],
        })
    return patient_list

X_all = create_patient_list(X_obs, ids, dt, cog, nhy)

# load saved EM run (theta_history, beta_history, beta_val if present)
saved_path = "/mnt/data/f_init_candidate_3_scld.npz"
z = np.load(saved_path, allow_pickle=True)

theta_history = z["theta_history"]        
beta_history = z["beta_history"]    
beta_val_saved = z["beta_val"] if "beta_val" in z.files else None

theta_vec = theta_history[:, -1]
f_star = theta_vec[:n_biomarkers]
s_star = theta_vec[n_biomarkers:2*n_biomarkers]
scalar_K_star = float(theta_vec[-1])
theta_star = {"f": f_star, "s": s_star, "scalar_K": scalar_K_star}

# rebuild the same split to align betas with patients
X_train, X_val = train_test_split(X_all, test_size=0.2, random_state=75)

ids_train = [p["id"] for p in X_train]
ids_val = [p["id"] for p in X_val]

beta_train = beta_history[:, -1]  # last iteration betas for train
pid_to_beta = {pid: float(b) for pid, b in zip(ids_train, beta_train)}

if beta_val_saved is not None and len(ids_val) == len(beta_val_saved):
    for pid, b in zip(ids_val, beta_val_saved.astype(float)):
        pid_to_beta[pid] = float(b)
else:
    # if no beta_val saved
    X_all = X_train
    ids_all = ids_train

# build subtyping payload with fixed betas and shared theta init
def create_patient_list_for_subtyping(X_obs, ids, dt, cog, nhy, beta, theta, biomarker_names=None):
    unique_ids = np.unique(ids)
    # expand to per-visit arrays
    ids_arr = np.asarray(ids)
    dt_arr = np.asarray(dt, float)
    X_arr = np.asarray(X_obs, float)
    cog_arr = np.asarray(cog, float)
    nhy_arr = np.asarray(nhy, float)

    # normalize theta
    theta_base = {"f": np.array(theta["f"], float),
                  "s": np.array(theta["s"], float),
                  "scalar_K": float(theta["scalar_K"])}
    
    theta_vec = np.concatenate([
    theta_base["f"],
    theta_base["s"],
    [theta_base["scalar_K"]]
    ])

    plist = []
    for pid in unique_ids:
        m = (ids_arr == pid)
        pdata = {
            "id": pid,
            "X_obs": X_arr[m],
            "dt": dt_arr[m],
            "beta": float(beta[pid]),
            "theta_init": theta_vec.copy(),
        }
        pdata["cog"] = cog_arr[m]
        pdata["nhy"] = nhy_arr[m]
        if biomarker_names is not None:
            pdata["biomarker_names"] = biomarker_names
        plist.append(pdata)
    return plist

# if you had both train and val betas:
ids_all = np.concatenate([ids_train, ids_val]) if len(pid_to_beta) > len(ids_train) else np.array(ids_train)
mask_all = np.isin(ids, ids_all)
X_subtype = create_patient_list_for_subtyping(
    X_obs=X_obs[mask_all],
    ids=ids[mask_all],
    dt=dt[mask_all],
    cog=cog[mask_all],
    nhy=nhy[mask_all],
    beta=pid_to_beta,
    theta=theta_star,
    biomarker_names=biomarker_names
)

# per-patient theta fits with fixed betas
subj = SubjectEM(K=K, verbose=1)
# re-use the same time grid you used in EM
# if not saved, rebuild a grid that comfortably covers dt + beta
t_min = 0.0
t_max = 40.0
n_steps = 4000
subj.t_span = np.linspace(t_min, t_max, n_steps)

subj.fit(X_subtype)
DeltaF = subj.transform()

# cluster and plot
res = cluster_and_plot(DeltaF,
                       n_clusters=3,
                       method="gmm",
                       use_pca=True,
                       pca_components=10,
                       standardize=True,
                       random_state=75,
                       out_path="/tmp/pca_clusters_from_npz.png")

labels = res["labels"]
probs = res["probs"]
print("cluster counts:", np.bincount(labels))
