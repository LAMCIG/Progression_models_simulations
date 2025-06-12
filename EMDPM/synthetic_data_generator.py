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
        t, x_true, _ = generate_logistic_model(n_biomarkers=n_biomarkers, t_max=t_max)

    beta_true_dict = {}
    X = []
    
    # TODO: add a better way to pass in lower and upper bound for regression param initialization.
    cog_a = float(rng.uniform(1,5,1))
    cog_b = float(rng.uniform(0,10,1))
    cog_noise =  float(rng.normal(0,1,1))
    
    print(f"a = {cog_a}, b = {cog_b}")

    for patient_id in range(n_patients):
        visit_interval = rng.gamma(shape=2, scale=0.5)
        first_visit = rng.uniform(0, t_max - (n_patient_obs * visit_interval))

        t_obs = np.array([first_visit + i * visit_interval for i in range(n_patient_obs)])
        t_obs = t_obs.clip(0, t_max)

        t_cumsum = np.insert(np.cumsum(np.diff(t_obs)), 0, 0)

        x_obs = x_true[:, np.searchsorted(t, t_obs)] + rng.normal(0, noise_level, (n_biomarkers, n_patient_obs))
        x_obs = x_obs.clip(0, 1)
        
        cognitive_scores = cog_a*(t_obs + rng.normal(0, 1, size=n_patient_obs)) + cog_b #+ cog_noise
        beta_true_dict[patient_id] = first_visit
        
        for i in range(n_patient_obs):
            X.append([patient_id, t_cumsum[i], cognitive_scores[i], beta_true_dict[patient_id]] + list(x_obs[:, i]))

    columns = ["patient_id", "dt", "cognitive_score", "beta_true"] + [f"biomarker_{i+1}" for i in range(n_biomarkers)]
    df = pd.DataFrame(X, columns=columns)

    return df, cog_a, cog_b

def initialize_beta(df: pd.DataFrame, beta_range: tuple = (0, 12), seed: int = 75, rng: np.random.Generator= None) -> pd.DataFrame:
    """
    Uniformly randomly initialize beta values for each patient ID.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing patient observations.
    beta_range : tuple
        Range to sample initial beta values from.

    Returns
    -------
    pd.DataFrame
        DataFrame with patient_id and initial beta column "0".
    """    
    if rng is None:
        rng = np.random.default_rng(seed)

    
    df = df.copy()
    patient_ids = df["patient_id"].unique()
    beta_values = rng.uniform(beta_range[0], beta_range[1], size=len(patient_ids))

    beta_map = dict(zip(patient_ids, beta_values))
    beta_column = df["patient_id"].map(beta_map)

    beta_iter = pd.DataFrame({"patient_id": df["patient_id"], "0": beta_column})
    return beta_iter
