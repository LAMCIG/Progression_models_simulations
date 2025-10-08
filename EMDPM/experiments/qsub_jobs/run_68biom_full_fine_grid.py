import numpy as np
import pandas as pd
from EMDPM.em_transformer_rework import EM

import statsmodels.formula.api as smf
from sklearn.model_selection import GroupKFold, KFold, GridSearchCV
from EMDPM.utils import initialize_f_eigen
import os

df = pd.read_csv("/data01/bgutman/MRI_data/PPMI/data_ppmi_pd.csv")
#df_K = pd.read_csv("/data01/bgutman/LEGACY/Skoltech/datasets/Connectomes/mean_NORM_con_22.csv")
#df_K = pd.read_csv("/data01/bgutman/LEGACY/Skoltech/datasets/Connectomes/mean_NORM_con_min_edges=3__sparsity=0.3.csv")
df_K = pd.read_csv("/data01/bgutman/LEGACY/Skoltech/datasets/Connectomes/mean_NORM_con_min_edges=3__sparsity=0.1.csv")

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

# def compute_patient_avg_slope(X, dt, ids):
#     unique_ids = np.unique(ids)
#     pid_to_slope = {}
    
#     for pid in unique_ids:
#         mask = (ids == pid)
#         t = dt[mask].reshape(-1,1)
#         Xi = X[mask]                       
#         if Xi.shape[0] < 2:
#             pid_to_slope[pid] = 0.0
#             continue
        
#         slopes = []
#         for j in range(Xi.shape[1]):
#             y = Xi[:, j]
#             # simple linear fit: biomarker_j = a * dt + b
#             a = np.linalg.lstsq(np.hstack([t, np.ones_like(t)]), y, rcond=None)[0][0]
#             slopes.append(a)
#         pid_to_slope[pid] = np.mean(slopes)
        
#     return pid_to_slope

# def prepare_glm_data(df, X, dt, ids, cog_df):
#     df_proc = df.copy()
#     df_proc['dt'] = dt
#     df_proc['id'] = ids
        
#     df_proc[['MCATOT','PIGD_score','TD_score']] = cog_df[['MCATOT','PIGD_score','TD_score']]
#     pid_to_slope = compute_patient_avg_slope(X, dt, ids)
#     df_proc['avg_slope'] = df_proc['id'].map(pid_to_slope)
    
#     print(df_proc['avg_slope'].describe())
    
#     return df_proc

# def init_beta_mixedlm(df_proc, t_max):
#     # Use dt_scaled as response; random intercept on 'id'
#     model = smf.mixedlm("dt ~ MCATOT + PIGD_score + TD_score + avg_slope",
#                          data=df_proc,
#                          groups=df_proc["id"],
#                          re_formula="1")
#     result = model.fit()
    
#     # intercepts
#     rand_eff = result.random_effects  # dict pid->{'Group': val}
#     pid_to_beta = {pid: eff["Group"] * t_max for pid, eff in rand_eff.items()}
#     return pid_to_beta

# df_proc = prepare_glm_data(df=df, X=X_obs, dt=dt, ids=ids, cog_df=df[["MCATOT","PIGD_score","TD_score"]])

# pid_to_beta = init_beta_mixedlm(df_proc, t_max=t_max)
# unique_ids = np.unique(ids)
# beta_init = np.array([pid_to_beta.get(pid, 0.0) for pid in unique_ids])

# initial_beta = (beta_init + np.abs(min(beta_init)))*1e9
# plt.hist(initial_beta)

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

param_grid = {
    "lambda_f": [0.92, 0.95, 0.97, 1.1],
    "lambda_cog": [1e-2, 1e-3],
    "lambda_scalar": [0.3, 0.50, 0.7],
    "jac_toggle": [True],
    "max_iter": [30],
    "t_max": [40],
    "epsilon": [1e-1],
}

groups_train = [p["id"] for p in X_train]

# 9/16/2025 group k fold made no sense

# grid = GridSearchCV(
#     estimator=EM(K=K),
#     param_grid=param_grid,
#     cv=GroupKFold(n_splits=3),
#     scoring=None,
#     n_jobs=28
# )
# grid.fit(X=X_train, y=None, groups=groups_train)

from sklearn.model_selection import KFold, GridSearchCV

cv = KFold(n_splits=3, shuffle=True, random_state=75)
grid = GridSearchCV(
    estimator=EM(K=K),
    param_grid=param_grid,
    cv=cv,
    scoring=None,     # uses EM.score()
    n_jobs=28
)
grid.fit(X=X_train, y=None)  # no groups needed

print("Best score:", grid.best_score_)
print("Best params:", grid.best_params_)

best_model = grid.best_estimator_
beta_val = best_model.transform(X_val)

out_path = "/home/dsemchin/Progression_models_simulations/EMDPM/experiments/qsub_jobs/qsub_results/em_paramsearch_K_sparse01.npz"

np.savez(out_path,
         theta_history=np.array(best_model.theta_history),
         cog_history=np.array(best_model.cog_regression_history),
         beta_history=np.array(best_model.beta_history),
         lse_history=np.array(best_model.lse_history),
         beta_val=np.array(beta_val),
         params=grid.best_params_)
