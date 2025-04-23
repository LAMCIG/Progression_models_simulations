import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# TODO: add general params for all plots

def plot_biomarker_trajectories(t: np.ndarray, x_true: np.ndarray, n_biomarkers: int = 10) -> None:
    """
    Plots ground truth biomarker trajectories over time.
    """
    colors = plt.cm.rainbow(np.linspace(0, 1, n_biomarkers))
    plt.figure(figsize=(10, 4))
    for i in range(n_biomarkers):
        plt.plot(t, x_true[i], label=f'biomarker {i+1}', color=colors[i])
    plt.title("biomarker trajectories")
    plt.legend()
    plt.show()

def plot_true_observations(df: pd.DataFrame, t: np.ndarray, x_true: np.ndarray, patient_idx=None) -> None:
    """
    Overlays observed biomarker values from a few patients on the true model trajectories.
    """
    if patient_idx is None:
        patient_idx = [0, 1, 2, 3, 4]
    n_biomarkers = x_true.shape[0]
    plt.figure(figsize=(10, 4))
    for i in range(n_biomarkers):
        plt.plot(t, x_true[i], color="k", alpha=0.5)
    patient_colors = plt.cm.rainbow(np.linspace(0, 1, len(patient_idx)))
    for color_index, patient in enumerate(patient_idx):
        patient_data = df[df["patient_id"] == patient]
        t_ij = patient_data["beta_true"] + patient_data["dt"]
        for i in range(n_biomarkers):
            plt.scatter(t_ij, patient_data[f"biomarker_{i+1}"], color=patient_colors[color_index], label=f"Patient {patient}" if i == 0 else None)
    plt.title("patient obs at true time")
    plt.legend()
    plt.show()

def plot_initial_beta_guess(df: pd.DataFrame, beta_iter: pd.DataFrame, t: np.ndarray, x_true: np.ndarray, patient_idx=None) -> None:
    """
    Visualizes initial guessed onset (beta) against ground truth onset.
    """
    if patient_idx is None:
        patient_idx = [0, 1, 2, 3, 4]
    n_biomarkers = x_true.shape[0]
    plt.figure(figsize=(10, 4))
    for i in range(n_biomarkers):
        plt.plot(t, x_true[i], color="k", alpha=0.2)
    patient_colors = plt.cm.rainbow(np.linspace(0, 1, len(patient_idx)))
    for color_index, patient in enumerate(patient_idx):
        beta_guess = beta_iter.loc[beta_iter["patient_id"] == patient, "0"].values[0]
        beta_true = df[df["patient_id"] == patient]["beta_true"].iloc[0]
        plt.axvline(x=beta_guess, color=patient_colors[color_index], linestyle="--", label=f"Patient {patient} pred")
        plt.axvline(x=beta_true, color=patient_colors[color_index], linestyle="-", label=f"Patient {patient} true")
    plt.title("true onset vs. initial onset prediction")
    plt.legend()
    plt.show()

def plot_theta_fit_comparison(t: np.ndarray, t_span: np.ndarray, x_true: np.ndarray, x_init: np.ndarray, x_final: np.ndarray, n_biomarkers: int) -> None:
    """
    Plots the initial, fitted, and true biomarker curves.
    """
    colors = plt.cm.rainbow(np.linspace(0, 1, n_biomarkers))
    plt.figure(figsize=(10, 4))
    for i in range(n_biomarkers):
        plt.plot(t, x_true[i], linestyle="-", color=colors[i], alpha=0.6)
        plt.plot(t_span, x_final[i], linestyle="--", color=colors[i], alpha=0.9)
        plt.plot(t_span, x_init[i], linestyle=":", color=colors[i], alpha=0.2)
    plt.title("intial vs. predicted vs. true curve")
    plt.show()

def plot_theta_error_history(theta_iter: pd.DataFrame, n_biomarkers: int, num_iterations: int, f_true: np.ndarray) -> None:
    """
    Plots the mean absolute error for x0 and f over iterations.
    """
    # x0_true = np.zeros(n_biomarkers)
    # f_true = np.zeros(n_biomarkers)
    # f_true[0] = 0.01
    #x0_error_history = []
    f_error_history = []
    for iteration in range(num_iterations):
        theta_column = f"iter_{iteration}"
        if theta_column in theta_iter.columns:
            #x0_estimated = theta_iter[theta_column].values[:n_biomarkers]
            #x0_error = np.mean(np.abs(x0_true - x0_estimated))
            #x0_error_history.append(x0_error)
            f_estimated = theta_iter[theta_column].values[n_biomarkers:]
            f_error = np.mean(np.abs(f_true - f_estimated)) / np.mean(f_true)
            f_error_history.append(f_error)
    plt.figure(figsize=(10, 4))
    #plt.plot(range(num_iterations), x0_error_history, label="x0 Error")
    plt.plot(range(num_iterations), f_error_history, label="f Error")
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.legend()
    plt.show()

def plot_beta_overlay(df: pd.DataFrame, beta_iter: pd.DataFrame, theta_iter: pd.DataFrame, t_span: np.ndarray, n_biomarkers: int, x_init: np.ndarray, x_final: np.ndarray, iteration: int = -1, patient_range=(0, 10)) -> None:
    """
    Plots vertical lines for beta guesses vs true across patients with fitted curve.
    """
    patients = df["patient_id"].unique()[patient_range[0]:patient_range[1]]
    beta_colors = plt.cm.rainbow(np.linspace(0, 1, len(patients)))
    plt.figure(figsize=(10, 5))
    for i in range(n_biomarkers):
        plt.plot(t_span, x_final[i], color="k", alpha=0.2)
    for idx, patient_id in enumerate(patients):
        beta_true = df[df["patient_id"] == patient_id]["beta_true"].iloc[0]
        beta_estimated = beta_iter.loc[beta_iter["patient_id"] == patient_id, str(iteration)].values[0]
        beta_init = beta_iter.loc[beta_iter["patient_id"] == patient_id, str(0)].values[0]
        plt.axvline(x=beta_true, color=beta_colors[idx], linestyle="-")
        plt.axvline(x=beta_estimated, color=beta_colors[idx], linestyle="--")
        plt.axvline(x=beta_init, color=beta_colors[idx], linestyle=":", alpha=0.3)
    plt.title("initial vs. predicted vs. true onset")
    plt.legend()
    plt.show()

def plot_beta_error_history(beta_iter: pd.DataFrame, df: pd.DataFrame, num_iterations: int) -> None:
    """
    Plots mean beta estimation error over EM iterations.
    """
    beta_error_history = []
    for iteration in range(1, num_iterations + 1):
        beta_column = str(iteration)
        if beta_column in beta_iter.columns:
            beta_estimated = beta_iter[["patient_id", beta_column]].rename(columns={beta_column: "beta_estimated"})
            beta_diff = df[["patient_id", "beta_true"]].merge(beta_estimated, on="patient_id")
            beta_diff["beta_error"] = np.abs(beta_diff["beta_true"] - beta_diff["beta_estimated"])
            beta_error_history.append(beta_diff["beta_error"].mean())
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(beta_error_history) + 1), beta_error_history)
    plt.xlabel("iteration")
    plt.ylabel("mean error")
    plt.show()
    
def plot_lse(lse_array: list) -> None:
    """
    Plots LSE trace across iterations.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(lse_array)), lse_array)
    plt.xlabel("iteration")
    plt.ylabel("LSE")
    plt.show()
