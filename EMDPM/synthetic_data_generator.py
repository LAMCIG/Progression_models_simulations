import pandas as pd
import numpy as np
import random
from .model_generator import generate_logistic_model

def generate_synthetic_data(n_biomarkers: int = 10, t_max: float = 12, noise_level: float = 0.0,
                            n_patients: int = 200, n_patient_obs: int = 3,
                            x_true: np.ndarray = None, t: np.ndarray = None,
                            seed: int = 75, rng: np.random.Generator = None) -> tuple:
    """
    Generate synthetic longitudinal patient data from a multivariate logistic model.

    Parameters
    ----------
    n_biomarkers : int
        Number of biomarkers per patient.
    t_max : float
        Maximum time span for simulation.
    noise_level : float
        Standard deviation of Gaussian noise added to biomarker observations.
    n_patients : int
        Number of synthetic patients to simulate.
    n_patient_obs : int
        Number of visits per patient.
    x_true : np.ndarray, optional
        Ground truth biomarker trajectories (optional override).
    t : np.ndarray, optional
        Corresponding time points for x_true (optional override).

    Returns
    -------
    tuple
        (df, beta_true_dict)
        df : pd.DataFrame
            Patient observation data.
        beta_true_dict : dict
            Ground truth beta values per patient.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    if x_true is None or t is None:
        t, x_true, _K, _x0, _f, _scalar_K = generate_logistic_model(n_biomarkers=n_biomarkers, t_max=t_max)

    # sanity: need room for n_patient_obs visits with unit spacing
    assert n_patient_obs * 1.0 < t_max, "t_max must be larger than n_patient_obs for unit visit spacing."

    X = []
    beta_true_dict = {}

    # simple cognitive model params
    cog_a = float(rng.uniform(1, 5, 1))
    cog_b = float(rng.uniform(0, 10, 1))
    print(f"a = {cog_a}, b = {cog_b}")

    visit_interval = 1.0
    max_first = t_max - n_patient_obs * visit_interval  # ensures last visit < t_max

    for patient_id in range(n_patients):
        first_visit = rng.uniform(0.0, max_first)

        # absolute observation times and relative times since first visit
        t_obs = first_visit + np.arange(n_patient_obs, dtype=float) * visit_interval
        dt_obs = np.arange(n_patient_obs, dtype=float)  # 0,1,2,...

        # sample biomarker values from trajectories
        idx = np.searchsorted(t, t_obs, side="left") #TODO: change to interp
        idx = np.clip(idx, 0, len(t) - 1)
        
        # x_true: (B, T) -> (B, n_visits); add noise and clip to [0,1]
        x_obs = x_true[:, idx] + rng.normal(0, noise_level, (n_biomarkers, n_patient_obs))
        x_obs = np.clip(x_obs, 0.0, 1.0)

        cognitive_scores = cog_a * (t_obs + rng.normal(0, 1, size=n_patient_obs)) + cog_b

        # one beta per patient: beta = first_visit
        beta_true = float(first_visit)
        beta_true_dict[patient_id] = beta_true
        

        for i in range(n_patient_obs):
            X.append(
                [patient_id, dt_obs[i], cognitive_scores[i], beta_true]
                + list(x_obs[:, i])
            )
            
    

    columns = ["patient_id", "dt", "cognitive_score", "beta_true"] + [f"biomarker_{i+1}" for i in range(n_biomarkers)]
    df = pd.DataFrame(X, columns=columns)

    return df, cog_a, cog_b


# def initialize_beta(df: pd.DataFrame, beta_range: tuple = (0, 12), seed: int = 75, rng: np.random.Generator= None) -> pd.DataFrame:
#     """
#     Uniformly randomly initialize beta values for each patient ID.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame containing patient observations.
#     beta_range : tuple
#         Range to sample initial beta values from.

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with patient_id and initial beta column "0".
#     """    
#     if rng is None:
#         rng = np.random.default_rng(seed)

    
#     df = df.copy()
#     patient_ids = df["patient_id"].unique()
#     beta_values = rng.uniform(beta_range[0], beta_range[1], size=len(patient_ids))

#     beta_map = dict(zip(patient_ids, beta_values))
#     beta_column = df["patient_id"].map(beta_map)

#     beta_iter = pd.DataFrame({"patient_id": df["patient_id"], "0": beta_column})
#     return beta_iter
