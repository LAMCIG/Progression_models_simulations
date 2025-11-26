import numpy as np
import sys
import os
from EMDPM.model_generator import generate_logistic_model, get_adjacency_matrix, create_patient_list
from EMDPM.synthetic_data_generator import generate_synthetic_data
from EMDPM.utils import *
from EMDPM.subtyping_em_transformer import SubtypingEM
from sklearn.model_selection import GridSearchCV, KFold

pbs_array_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
# Fixed seed for data generation (same across all PBS jobs)
data_seed = 75
# Different seed for EM initialization (varies per PBS job)
em_seed = 75 + pbs_array_id * 100

# Data generation parameters
n_biomarkers = 3
t_max = 25
noise_level = 0.0
n_patients = 150
n_patient_obs = 3
n_subtypes = 2

print(f"PBS Array ID: {pbs_array_id}, Data seed: {data_seed}, EM seed: {em_seed}")


K = get_adjacency_matrix("random_offdiag", n_biomarkers, np.random.RandomState(data_seed))

# True parameters for each subtype
scalar_K_list = [0.2, 0.2]
f_list = [np.array([0.0, 0.0, 0.3]), np.array([0.2, 0.0, 0.0])]

# Generate synthetic data
X = []
y = []

for subtype in range(n_subtypes):
    t, x_true, K, x0, f, scalar_K = generate_logistic_model(
        n_biomarkers=n_biomarkers,
        scalar_K=scalar_K_list[subtype],
        t_max=t_max,
        K=K,
        f=f_list[subtype]
    )
    
    df, cog_a, cog_b = generate_synthetic_data(
        n_biomarkers=n_biomarkers,
        t_max=t_max,
        noise_level=noise_level,
        n_patients=n_patients,
        n_patient_obs=n_patient_obs,
        x_true=x_true,
        t=t,
        rng=np.random.RandomState(data_seed + subtype + 2),  # FIXED seed
    )
    
    biomarker_cols = [col for col in df.columns if col.startswith("biomarker_")]
    X_obs = df[biomarker_cols].values
    dt = df["dt"].values
    ids = df["patient_id"].values + subtype * n_patients
    cog = df["cognitive_score"].values
    beta_true = df["beta_true"].values
    
    X_subtype = create_patient_list(X_obs, ids, dt, cog, initial_beta=None)
    
    for p in X_subtype:
        p["subtype_true"] = int(subtype)
    X += X_subtype
    
    y.append(beta_true[::n_patient_obs])

X = np.asarray(X)
y = np.concatenate(y)

# Initialize beta
all_dt = np.concatenate([p["dt"] for p in X])
all_cog = np.concatenate([p["cog"] for p in X])
all_ids_array = np.concatenate([[p["id"]] * len(p["dt"]) for p in X])

initial_beta, pid_to_beta, result = fit_mixedlm_beta_from_clinical(
    ids=all_ids_array,
    dt=all_dt,
    cog=all_cog,
    t_max=t_max,
    verbose=False,
    rng=np.random.default_rng(data_seed)  # FIXED seed
)

# Add initial_beta to patient data
unique_ids = np.unique([p["id"] for p in X])
id_to_beta_idx = {pid: idx for idx, pid in enumerate(unique_ids)}
for p in X:
    p["initial_beta"] = initial_beta[id_to_beta_idx[p["id"]]]

# Initialize f with random seed (DIFFERENT for each PBS array job)
rng_f = np.random.RandomState(em_seed + 500)
f_init = initialize_f_eigen(K=K, rng=rng_f)
if isinstance(f_init, list):
    f_init = f_init[0]

# Don't set initial_assignments here - let the model generate them internally
# based on the rng seed, so they match the training set size during CV

# Parameter grid for gridsearch
param_grid = {
    "lambda_f": [0.1, 0.25, 0.5, 0.75, 1.0],
    "lambda_cog": [0.01, 0.025, 0.05, 0.1],
    "lambda_scalar": [0.01, 0.05, 0.1, 0.25, 0.5],
    "jac_toggle": [True],
    "max_iter": [100],
    "t_max": [t_max],
    "epsilon": [1e-1, 1e-2],
}

# Create model with base parameters
# The model will generate random assignments internally based on rng seed
# Each PBS array job runs ONE gridsearch with different random EM initialization
# (but same ground truth data)
model = SubtypingEM(
    K=K,
    initial_f=f_init,
    n_subtypes=n_subtypes,
    initial_assignments=None,  # Let model generate based on rng
    verbose=0,
    rng=np.random.default_rng(em_seed)  # Different EM seed per PBS job
)

# Run gridsearch with KFold cross-validation
# Each PBS array job (0-9) runs one complete gridsearch with different random EM initialization
# All jobs use the SAME ground truth data for fair comparison
print("Starting GridSearchCV...")
print(f"Running gridsearch with {len(param_grid['lambda_f']) * len(param_grid['lambda_cog']) * len(param_grid['lambda_scalar']) * len(param_grid['epsilon'])} parameter combinations")
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=KFold(n_splits=3, shuffle=True, random_state=data_seed),  # FIXED seed for CV splits
    scoring=None,  # Uses estimator's score method (negative LSE)
    n_jobs=1,  # Each PBS job runs one gridsearch sequentially
    verbose=1
)

grid.fit(X=X, y=None)

print(f"\nBest score (negative LSE): {grid.best_score_}")
print(f"Best params: {grid.best_params_}")

# Get best model and compute final LSE
best_model = grid.best_estimator_
final_lse = best_model.lse_history[-1] if hasattr(best_model, 'lse_history') else None

# Save results
output_dir = "/home/dsemchin/Progression_models_simulations/EMDPM/experiments/qsub_jobs/qsub_results"
os.makedirs(output_dir, exist_ok=True)

out_path = os.path.join(output_dir, f"synthetic_gridsearch_array_{pbs_array_id}.npz")

np.savez(
    out_path,
    best_params=grid.best_params_,
    best_score=grid.best_score_,
    final_lse=final_lse,
    pbs_array_id=pbs_array_id,
    data_seed=data_seed,
    em_seed=em_seed,
    cv_results=grid.cv_results_,
    beta_history=best_model.beta_history if hasattr(best_model, 'beta_history') else None,
    lse_history=best_model.lse_history if hasattr(best_model, 'lse_history') else None,
    # Ground truth parameters (same for all PBS jobs)
    f_true_list=f_list,
    scalar_K_true_list=scalar_K_list,
    beta_true_array=y,
    K=K,
)

print(f"Results saved to {out_path}")
