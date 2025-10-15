import numpy as np
import pandas as pd
import os
import copy
from EMDPM.em_transformer_rework import EM
from EMDPM.utils import initialize_f_eigen
from EMDPM.subject_EM import SubjectEM
from EMDPM.subject_subtyping import *

import statsmodels.formula.api as smf
from sklearn.model_selection import GroupKFold, GridSearchCV

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--candidate", type=int, default=0)
args = parser.parse_args()
current_candidate = 0

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

def _z(x):
    x = np.asarray(x, float)
    m = np.nanmean(x); s = np.nanstd(x)
    return (x - m) / (s if np.isfinite(s) and s > 0 else 1.0)

def build_severity_index(df):
    moca = _z(df["MCATOT"].to_numpy())        # higher is better
    td   = _z(df["TD_score"].to_numpy())
    pigd = _z(df["PIGD_score"].to_numpy())
    S = (-moca + td + pigd) / 3.0             # higher = worse / later
    return S

def fit_mixedlm_beta_from_clinical(df, ids, dt, t_max, verbose=False, rng=None):

    if rng is None:
        rng = np.random.default_rng(0)

    S = build_severity_index(df)
    dfm = pd.DataFrame({"id": ids, "dt": dt.astype(float), "S": S})
    # keep longitudinal subjects
    good_ids = dfm.groupby("id").size().pipe(lambda s: s[s >= 2]).index
    dfm = dfm[dfm["id"].isin(good_ids)].copy()

    # clean + tiny jitter on dt to avoid exact duplicate rows
    dfm = dfm.replace([np.inf, -np.inf], np.nan).dropna(subset=["id", "dt", "S"])
    dfm["dt"] = dfm["dt"] + 1e-6 * rng.standard_normal(len(dfm))

    # fit: severity ~ dt + (1|id)
    model = smf.mixedlm("S ~ dt", data=dfm, groups=dfm["id"], re_formula="1")
    result = None
    for meth in ("bfgs", "nm"):
        result = model.fit(method=meth, reml=True, maxiter=500, disp=False)
        break

    # get slope k (fixed effect of dt)
    if result is not None and result.converged:
        k = float(result.params.get("dt", np.nan))
    else:
        # fallback OLS for slope if MixedLM fails
        ols = smf.ols("S ~ dt", data=dfm).fit()
        k = float(ols.params.get("dt", np.nan))

    if not np.isfinite(k) or abs(k) < 1e-8:
        # if slope is (near) zero, default to small positive slope to avoid blow-up
        if verbose:
            print("[MixedLM] slope near zero; using fallback k=1.0")
        k = 1.0

    # random intercepts u_i
    re = {}
    if result is not None and hasattr(result, "random_effects"):
        re = {pid: float(eff.get("Group", 0.0)) for pid, eff in result.random_effects.items()}

    unique_ids = np.unique(ids)
    beta_raw = np.array([re.get(pid, 0.0) / k for pid in unique_ids], dtype=float)

    # shift & clip: make the smallest beta zero, cap at t_max
    beta_shift = beta_raw - np.nanmin(beta_raw)
    initial_beta = np.clip(beta_shift, 0.0, t_max)

    pid_to_beta = {pid: initial_beta[i] for i, pid in enumerate(unique_ids)}

    if verbose:
        print("beta_init summary:", pd.Series(initial_beta).describe())

    return initial_beta, pid_to_beta, result


initial_beta, pid_to_beta, _ = fit_mixedlm_beta_from_clinical(df=df, 
                                                              ids=ids,
                                                              dt=dt, 
                                                              t_max=t_max, 
                                                              verbose=True
                                                              )



### INITIALIZE F
f_init_list = initialize_f_eigen(K=K, 
                                 jitter_strength=0.05, 
                                 n_eigs=100,
                                 rng=np.random.RandomState(75)
                                 )

f_init = f_init_list[0]


f_rand_list = []
for i in range(100):
    f_rand_list.append(np.random.uniform(0.0, 0.3, n_biomarkers))
    
#f_init = f_rand_list[current_candidate]


### DATA SPLIT

from sklearn.model_selection import train_test_split

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

model = EM(K=K)

em = EM(K=K,
        lambda_f=1.0,
        lambda_cog=0.01,
        lambda_scalar=0.3,
        initial_f=f_init,
        jac_toggle=True,
        max_iter=30,
        t_max=40,
        epsilon=1e-1)

em.fit(X_train)
beta_val = em.transform(X_val)

out_dir = "/home/dsemchin/Progression_models_simulations/EMDPM/experiments/qsub_jobs/qsub_results"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"f_init_glm_eigen_sparse03_{current_candidate}.npz")

np.savez(out_path,
         theta_history=np.array(em.theta_history),
         cog_history=np.array(em.cog_regression_history),
         beta_history=np.array(em.beta_history),
         lse_history=np.array(em.lse_history),
         beta_val=np.array(beta_val),
         candidate=current_candidate,
         f_init=f_init)
print("Saved:", out_path)

ids_train = [p["id"] for p in X_train]
ids_val   = [p["id"] for p in X_val]

# final betas after fit for the training patients:
beta_train = np.asarray(em.beta_, dtype=float)  # aligned with ids_train

# betas for validation patients returned by transform:
beta_val = np.asarray(beta_val, dtype=float)    # aligned with ids_val

# combine into a single pid -> beta mapping
pid_to_beta = {pid: float(b) for pid, b in zip(ids_train, beta_train)}
pid_to_beta.update({pid: float(b) for pid, b in zip(ids_val, beta_val)})

def create_patient_list_for_subtyping(
    X_obs,
    ids,
    dt,
    cog=None,
    nhy=None,
    beta=None,
    theta=None,
    biomarker_names=None,
):


    # stable, first-seen order of patients
    unique_ids = pd.unique(ids)

    # normalize beta input to a pid->beta dict
    if isinstance(beta, dict):
        pid_to_beta = beta
    else:
        if beta is None:
            raise ValueError("beta must be provided (dict or ndarray).")
        if len(beta) != len(unique_ids):
            raise ValueError(
                f"beta length {len(beta)} does not match number of unique ids {len(unique_ids)}."
            )
        pid_to_beta = {pid: float(beta[i]) for i, pid in enumerate(unique_ids)}

    # normalize theta to a plain dict, then deep-copy per patient
    def _theta_to_dict(th):
        if th is None:
            raise ValueError("theta must be provided.")
        if isinstance(th, dict):
            return {"f": np.array(th["f"], float),
                    "s": np.array(th["s"], float),
                    "scalar_K": float(th["scalar_K"])}
        # object with attributes
        out = {}
        for key in ("f", "s", "scalar_K"):
            if not hasattr(th, key):
                raise ValueError(f"theta missing attribute '{key}'.")
        out["f"] = np.array(getattr(th, "f"), float)
        out["s"] = np.array(getattr(th, "s"), float)
        out["scalar_K"] = float(getattr(th, "scalar_K"))
        return out

    theta_base = _theta_to_dict(theta)

    # build list
    ids = np.asarray(ids)
    dt = np.asarray(dt, float)
    X_obs = np.asarray(X_obs, float)
    if cog is not None:
        cog = np.asarray(cog, float)
    if nhy is not None:
        nhy = np.asarray(nhy, float)

    patient_list = []
    for pid in unique_ids:
        mask = (ids == pid)
        pdata = {
            "id": pid,
            "X_obs": X_obs[mask],
            "dt": dt[mask],
            "beta": float(pid_to_beta[pid]),
            "theta_init": copy.deepcopy(theta_base),
        }
        if cog is not None:
            pdata["cog"] = cog[mask]
        if nhy is not None:
            pdata["nhy"] = nhy[mask]
        if biomarker_names is not None:
            pdata["biomarker_names"] = biomarker_names
        patient_list.append(pdata)

    return patient_list

theta_vec = em.theta
f_star = theta_vec[:n_biomarkers]
s_star = theta_vec[n_biomarkers:2*n_biomarkers]
scalar_K_star = theta_vec[-1]
theta_star = {"f": f_star, "s": s_star, "scalar_K": float(scalar_K_star)}

X_subtype = create_patient_list_for_subtyping(
    X_obs=X_obs, ids=ids, dt=dt, cog=cog, nhy=nhy,
    beta=pid_to_beta, theta=theta_star, biomarker_names=biomarker_names
)


subj = SubjectEM(K=K, verbose=1)
subj.t_span = em.t_span
subj.fit(X_subtype)

DeltaF = subj.transform()


res = cluster_and_plot(DeltaF,
                       n_clusters=3, 
                       method="gmm",    
                       use_pca=True,
                       pca_components=10,
                       standardize=True,
                       random_state=75,
                       out_path="/tmp/pca_clusters.png")

labels = res["labels"]
probs = res["probs"]  # None if method=kmeans
