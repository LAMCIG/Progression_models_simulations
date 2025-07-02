import numpy as np
import pandas as pd
from EMDPM.em_transformer_rework import EM

import statsmodels.formula.api as smf
from sklearn.model_selection import GroupKFold, GridSearchCV

df = pd.read_csv("/data01/bgutman/MRI_data/PPMI/data_ppmi_pd.csv")
df_K = pd.read_csv("/data01/bgutman/LEGACY/Skoltech/datasets/Connectomes/mean_NORM_con_22.csv")
n_biomarkers = 10

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

small_region_set = ['L_inferiorparietal_thickavg','R_inferiorparietal_thickavg','L_inferiortemporal_thickavg','R_inferiortemporal_thickavg',
'L_precuneus_thickavg','R_precuneus_thickavg','L_isthmuscingulate_thickavg','R_isthmuscingulate_thickavg','L_lateraloccipital_thickavg','R_lateraloccipital_thickavg']

small_region_set_resid = ['L_inferiorparietal_thickavg_resid','R_inferiorparietal_thickavg_resid','L_inferiortemporal_thickavg_resid','R_inferiortemporal_thickavg_resid',
'L_precuneus_thickavg_resid','R_precuneus_thickavg_resid','L_isthmuscingulate_thickavg_resid','R_isthmuscingulate_thickavg_resid','L_lateraloccipital_thickavg_resid','R_lateraloccipital_thickavg_resid']

#X = df[[col for col in df.columns if (col.startswith(('L_', 'R_')) and col.endswith('_thickavg') and not col.endswith('_thickavg_resid'))]]
X_obs_resid = df[small_region_set_resid]
X_obs_resid = X_obs_resid.to_numpy()
X_obs_resid = np.max(X_obs_resid, axis=0) - X_obs_resid

X_obs = df[small_region_set]
X_obs = X_obs.to_numpy()
X_obs = np.max(X_obs, axis=0) - X_obs

print("nans in X:", np.isnan(X_obs).sum())
print("infs in X:", np.isinf(X_obs).sum())

print("nans X_resid:", np.isnan(X_obs_resid).sum())
print("infs X_resid:", np.isinf(X_obs_resid).sum())

## connectivity matrix to numpy
K = df_K.drop(df_K.columns[0], axis=1).to_numpy()
np.fill_diagonal(K, 0)
print(K.shape, type(K))

# normalization
row_sums = K.sum(axis=1)
median_row_sum = np.median(row_sums)
K = K / median_row_sum

region_index = [6, 40, 7, 41, 23, 57, 8, 42, 9, 43]
K = np.take(np.take(K,region_index,axis=0), region_index, axis=1)

t_max = 30
step = 0.001
print("X.size: ", X_obs.shape, "X_resid.size: ", X_obs_resid.shape)

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

# plt.scatter(grouped["NSD_STAGE"], grouped["NHY"])
# plt.xlabel("Mean NSD Stage (per patient)")
# plt.ylabel("Mean NHY Score (per patient)")
# plt.title("NSD vs NHY (Patient-Level Averages)")
# plt.grid(True)
# plt.show()

nsd = df["NSD_STAGE"].to_numpy()

def compute_patient_avg_slope(X, dt, ids):
    unique_ids = np.unique(ids)
    pid_to_slope = {}
    
    for pid in unique_ids:
        mask = (ids == pid)
        t = dt[mask].reshape(-1,1)
        Xi = X[mask]                       
        if Xi.shape[0] < 2:
            pid_to_slope[pid] = 0.0
            continue
        
        slopes = []
        for j in range(Xi.shape[1]):
            y = Xi[:, j]
            # simple linear fit: biomarker_j = a * dt + b
            a = np.linalg.lstsq(np.hstack([t, np.ones_like(t)]), y, rcond=None)[0][0]
            slopes.append(a)
        pid_to_slope[pid] = np.mean(slopes)
        
    return pid_to_slope

def prepare_glm_data(df, X, dt, ids, cog_df):
    df_proc = df.copy()
    df_proc['dt'] = dt
    df_proc['id'] = ids
        
    df_proc[['MCATOT','PIGD_score','TD_score']] = cog_df[['MCATOT','PIGD_score','TD_score']]
    pid_to_slope = compute_patient_avg_slope(X, dt, ids)
    df_proc['avg_slope'] = df_proc['id'].map(pid_to_slope)
    
    print(df_proc['avg_slope'].describe())
    
    return df_proc

def init_beta_mixedlm(df_proc, t_max):
    # Use dt_scaled as response; random intercept on 'id'
    model = smf.mixedlm("dt ~ MCATOT + PIGD_score + TD_score + avg_slope",
                         data=df_proc,
                         groups=df_proc["id"],
                         re_formula="1")
    result = model.fit()
    
    # intercepts
    rand_eff = result.random_effects  # dict pid->{'Group': val}
    pid_to_beta = {pid: eff["Group"] * t_max for pid, eff in rand_eff.items()}
    return pid_to_beta

df_proc = prepare_glm_data(df=df, X=X_obs, dt=dt, ids=ids, cog_df=df[["MCATOT","PIGD_score","TD_score"]])

pid_to_beta = init_beta_mixedlm(df_proc, t_max=t_max)
unique_ids = np.unique(ids)
beta_init = np.array([pid_to_beta.get(pid, 0.0) for pid in unique_ids])

initial_beta = (beta_init + np.abs(min(beta_init)))*1.5e9
# plt.hist(initial_beta)

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

from sklearn.metrics import make_scorer

model = EM(K=K)
# param_grid = {
#     "lambda_f": [0.9, 0.95, 1.0, 1.1],
#     "lambda_cog": [0.01, 0.025, 0.05],
#     "lambda_scalar": [0.0, 0.01, 0.02],
#     "jac_toggle": [True],
#     "max_iter": [15],
#     "t_max": [40],
#     "epsilon": [1e-1],
# }

param_grid = {
    "lambda_f": [0.5, 0.7, 0.95],
    "lambda_cog": [0.01, 0.025, 0.05],
    "lambda_scalar": [0.0, 0.01, 0.05],
    "jac_toggle": [True],
    "max_iter": [20],
    "t_max": [40],
    "epsilon": [1e-1],
}

groups_train = [p["id"] for p in X_train]

grid = GridSearchCV(
    estimator=EM(K=K),
    param_grid=param_grid,
    cv=GroupKFold(n_splits=3),
    scoring=None,
    n_jobs=28
)
grid.fit(X=X_train, y=None, groups=groups_train)

print("Best score:", grid.best_score_)
print("Best params:", grid.best_params_)

import os

best_model = grid.best_estimator_
beta_val = best_model.transform(X_val)

out_path = "/home/dsemchin/Progression_models_simulations/EMDPM/experiments/qsub_jobs/qsub_results/em_best_model_10biom.npz"

np.savez(out_path,
         theta_history=np.array(best_model.theta_history),
         cog_history=np.array(best_model.cog_regression_history),
         beta_history=np.array(best_model.beta_history),
         lse_history=np.array(best_model.lse_history),
         beta_val=np.array(beta_val),
         params=grid.best_params_)
