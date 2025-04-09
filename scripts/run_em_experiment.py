import os
import pickle
import numpy as np
import pandas as pd
from EMDPM.model_generator import generate_logistic_model
from EMDPM.synthetic_data_generator import generate_synthetic_data
from EMDPM.em_transformer import EM
from EMDPM.utils import reconstruct_trajectories
from EMDPM.visualization import (
    plot_theta_fit_comparison,
    plot_theta_error_history,
    plot_beta_overlay,
    plot_beta_error_history
)
import argparse

def run_experiment(n_biomarkers: int, noise_level: float, use_jacobian: bool, t_max: int = 12, n_patients: int = 200, n_patient_obs: int = 3, num_iterations: int = 50):
    t_span = np.linspace(0, t_max, 2000)
    t, x_true, K = generate_logistic_model(n_biomarkers=n_biomarkers, t_max=t_max)
    df, beta_true_dict = generate_synthetic_data(
        n_biomarkers=n_biomarkers,
        t_max=t_max,
        noise_level=noise_level,
        n_patients=n_patients,
        n_patient_obs=n_patient_obs,
        x_true=x_true,
        t=t
    )

    em_model = EM(K=K, num_iterations=num_iterations, t_max=t_max, step=0.01, use_jacobian=use_jacobian)
    em_model.fit(df)

    x0_init = em_model.theta_iter_["iter_0"].values[:n_biomarkers]
    f_init = em_model.theta_iter_["iter_0"].values[n_biomarkers:]
    x0_final = em_model.theta_iter_[f"iter_{num_iterations-1}"].values[:n_biomarkers]
    f_final = em_model.theta_iter_[f"iter_{num_iterations-1}"].values[n_biomarkers:]
    x_init = reconstruct_trajectories(x0_init, f_init, K, t_span)
    x_final = reconstruct_trajectories(x0_final, f_final, K, t_span)

    result_dict = {
        "n_biomarkers": n_biomarkers,
        "noise_level": noise_level,
        "use_jacobian": use_jacobian,
        "em_model": em_model,
        "x_true": x_true,
        "x_init": x_init,
        "x_final": x_final,
        "t": t,
        "t_span": t_span,
        "df": df
    }

    output_name = f"em_results_b{n_biomarkers}_n{noise_level}_jac{use_jacobian}.pkl"
    output_path = os.path.join("results", output_name)
    with open(output_path, "wb") as f:
        pickle.dump(result_dict, f)

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EM fitting experiment with different settings.")
    parser.add_argument("--n_biomarkers", type=int, required=True)
    parser.add_argument("--noise_level", type=float, required=True)
    parser.add_argument("--use_jacobian", type=lambda x: x.lower() == 'true', required=True)
    args = parser.parse_args()

    run_experiment(
        n_biomarkers=args.n_biomarkers,
        noise_level=args.noise_level,
        use_jacobian=args.use_jacobian
    )
