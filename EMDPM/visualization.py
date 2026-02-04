import math
from typing import Optional, Sequence

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

# TODO: add general params for all plots

def plot_biomarker_trajectories(biom_trajectories: np.ndarray, t_span: np.ndarray, n_biomarkers: int = 68) -> None:
    """
    Plots ground truth biomarker trajectories over time.
    """
    plt.figure(figsize=(4,3))
    # Choose colormap based on number of biomarkers
    if n_biomarkers <= 10:
        cmap = plt.get_cmap("tab10")
    elif n_biomarkers <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.cm.rainbow
        colors = cmap(np.linspace(0, 1, n_biomarkers))
    
    for b in range(n_biomarkers):
        if n_biomarkers > 20:
            color = colors[b]
        else:
            color = cmap(b % cmap.N)
        plt.plot(t_span, biom_trajectories[b], color=color)
    plt.title("biomarker trajectories")
    plt.legend()
    plt.show()

def plot_true_observations(df: pd.DataFrame, t: np.ndarray, x_true: np.ndarray, patient_idx=None) -> None:
    """
    Overlays observed biomarker values from selected patients on the true model trajectories.
    Each biomarker gets a unique color, and each patient gets a unique marker shape.
    Marker colors match their corresponding biomarker trajectory colors.
    """
    n_biomarkers = x_true.shape[0]

    # Choose colormap based on number of biomarkers
    if n_biomarkers <= 10:
        cmap = plt.get_cmap("tab10")
    elif n_biomarkers <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("viridis")

    # Define marker styles
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*', '<', '>']

    fig, ax = plt.subplots(figsize=(10, 3))

    # Plot colored true trajectories
    for i in range(n_biomarkers):
        color = cmap(i % cmap.N)
        ax.plot(t, x_true[i], color=color, alpha=0.9, linewidth=1.5, label=f"Biomarker {i+1}")

    # Plot observations if patient_idx provided
    if patient_idx is not None:
        for m_idx, patient in enumerate(patient_idx):
            patient_data = df[df["patient_id"] == patient]
            t_ij = patient_data["beta_true"] + patient_data["dt"]
            marker = markers[m_idx % len(markers)]

            for i in range(n_biomarkers):
                y = patient_data[f"biomarker_{i+1}"]
                biomarker_color = cmap(i % cmap.N)
                ax.scatter(
                    t_ij,
                    y,
                    color=biomarker_color,
                    marker=marker,
                    s=30,
                    edgecolors="white",
                    linewidths=0.5,
                    label=f"Patient {patient}" if i == 0 else None,
                )

    ax.set_title("Synthetic patient observations on ground-truth biomarker trajectories")
    ax.set_xlabel("Time")
    ax.set_ylabel("Biomarker Value")
    ax.legend(fontsize=8, ncol=3, frameon=False)
    plt.tight_layout()
    plt.show()

def plot_beta_overlay(df: pd.DataFrame, beta_iter: pd.DataFrame, theta_iter: pd.DataFrame, t_span: np.ndarray,
                      n_biomarkers: int, x_init: np.ndarray, x_final: np.ndarray, iteration: int = -1,
                      patient_range=(0, 5)) -> None:
    """
    plots vertical lines for initial, predicted, and true beta (onset) per patient,
    along with ground truth biomarker trajectories.
    """
    patients = df["patient_id"].unique()[patient_range[0]:patient_range[1]]
    tab10 = plt.get_cmap("tab10")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # plot ground truth biomarker trajectories
    for i in range(n_biomarkers):
        ax.plot(t_span, x_final[i], color="k", alpha=0.2)

    for idx, patient_id in enumerate(patients):
        color = tab10(idx % 10)
        beta_true = df[df["patient_id"] == patient_id]["beta_true"].iloc[0]
        beta_estimated = beta_iter.loc[beta_iter["patient_id"] == patient_id, str(iteration)].values[0]
        beta_init = beta_iter.loc[beta_iter["patient_id"] == patient_id, "0"].values[0]

        ax.axvline(x=beta_true, color=color, linestyle="-", label=f"true (P{patient_id})")
        ax.axvline(x=beta_estimated, color=color, linestyle="--", label=f"est (P{patient_id})")
        # ax.axvline(x=beta_init, color=color, linestyle=":", alpha=0.3, label=f"init (P{patient_id})")
    
    # reduce legend duplicates by using handles/labels dict
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_title("Initial vs. Predicted vs. True Onset (Beta) per Patient")
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
        plt.plot(t, x_true[i], color="k", alpha=0.3)
    patient_colors = plt.cm.rainbow(np.linspace(0, 1, len(patient_idx)))
    for color_index, patient in enumerate(patient_idx):
        beta_guess = beta_iter.loc[beta_iter["patient_id"] == patient, "0"].values[0]
        beta_true = df[df["patient_id"] == patient]["beta_true"].iloc[0]
        plt.axvline(x=beta_guess, color=patient_colors[color_index], linestyle="--", label=f"Patient {patient} pred")
        plt.axvline(x=beta_true, color=patient_colors[color_index], linestyle="-", label=f"Patient {patient} true")
    plt.title("true onset vs. initial onset prediction")
    plt.legend()
    plt.show()

def plot_theta_fit_comparison(t: np.ndarray, t_span: np.ndarray, x_true: np.ndarray,
                              x_init: np.ndarray, x_final: np.ndarray,
                              n_biomarkers: int) -> None:
    """
    Plots the fitted vs. true biomarker trajectories for comparison.
    """
    if n_biomarkers == 10:
        cmap = plt.get_cmap("tab10")
    elif n_biomarkers == 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("viridis") # ill comeup with a better one later

    plt.figure(figsize=(10, 4))
    for i in range(n_biomarkers):
        color = cmap(i % cmap.N)
        plt.plot(t, x_true[i], linestyle="-", color=color, alpha=0.6,
                 label=f"Biomarker {i+1}" if n_biomarkers <= 10 else None)
        plt.plot(t_span, x_final[i], linestyle="--", color=color, alpha=0.9)
        #plt.plot(t_span, x_init[i], linestyle=":", color=color, alpha=0.2)
    # if n_biomarkers <= 10:
    #     plt.legend(ncol=2, fontsize="small", loc="upper left")

    plt.title("Predicted vs. true trajectories")
    plt.xlabel("Time")
    plt.ylabel("Biomarker Value")
    #plt.tight_layout()
    plt.show()

def plot_theta_error_history(theta_iter: pd.DataFrame, n_biomarkers: int, num_iterations: int,
                             f_true: np.ndarray, s_true: np.ndarray, scalar_K_true: float) -> None:
    """
    Plots normalized error of each parameter group over EM iterations.
    """
    f_error_history = []
    s_error_history = []
    scalar_K_error_history = []

    for iteration in range(num_iterations):
        theta_column = f"iter_{iteration}"
        if theta_column not in theta_iter.columns:
            continue

        theta = theta_iter[theta_column].values

        f_est = theta[n_biomarkers:2*n_biomarkers]
        a_est = theta[2*n_biomarkers:3*n_biomarkers]
        b_est = theta[3*n_biomarkers:4*n_biomarkers]
        scalar_K_est = theta[-1]  # or theta[4*n_biomarkers]

        f_err = np.mean(np.abs(f_true - f_est)) / (np.mean(np.abs(f_true)) + 1e-8)
        s_err = np.mean(np.abs(s_true - b_est)) / (np.mean(np.abs(s_true)) + 1e-8)
        k_err = np.abs(scalar_K_true - scalar_K_est) / (np.abs(scalar_K_true) + 1e-8)

        f_error_history.append(f_err)
        s_error_history.append(s_err)
        scalar_K_error_history.append(k_err)
        
    plt.figure(figsize=(10, 4))
    plt.plot(f_error_history, label="forcing error")
    plt.plot(s_error_history, label="supremum error")
    plt.plot(scalar_K_error_history, label="scalar error")
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.legend()
    plt.show()

def plot_beta_overlay(df: pd.DataFrame, beta_iter: pd.DataFrame, theta_iter: pd.DataFrame, t_span: np.ndarray,
                      n_biomarkers: int, x_init: np.ndarray, x_final: np.ndarray, iteration: int = -1, patient_range=(0, 5)) -> None:
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
    for iteration in range(num_iterations):
        beta_column = str(iteration)
        if beta_column in beta_iter.columns:
            beta_estimated = beta_iter[["patient_id", beta_column]].rename(columns={beta_column: "beta_estimated"})
            beta_diff = df[["patient_id", "beta_true"]].merge(beta_estimated, on="patient_id")
            beta_diff["beta_error"] = np.abs(beta_diff["beta_true"] - beta_diff["beta_estimated"])
            beta_error_history.append(beta_diff["beta_error"].mean())
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(beta_error_history)), beta_error_history)
    plt.ylim([0, max(beta_error_history)])
    plt.xlabel("iteration")
    plt.ylabel("mean beta error")
    plt.show()


def plot_patient_beta_histogram(beta_values: np.ndarray, bins: int = 30,
                                title: str = "Distribution of patient betas") -> None:
    """
    Plot a histogram of patient-specific beta values.
    """
    if beta_values is None or len(beta_values) == 0:
        raise ValueError("beta_values must be a non-empty array.")

    plt.figure(figsize=(8, 4))
    plt.hist(beta_values, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.8)
    plt.title(title)
    plt.xlabel("beta")
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_beta_history_by_subtype(
    beta_history: np.ndarray,
    assignments: np.ndarray,
    n_subtypes: int,
    beta_true: np.ndarray,
    subtype_labels: Optional[Sequence[str]] = None,
    subtype_mapping: Optional[np.ndarray] = None,
) -> None:
    """
    Plot mean absolute error (|beta_pred - beta_true|) over iterations by subtype with std bands,
    a histogram comparing final beta vs true beta, and a histogram of final beta by subtype.

    Parameters
    ----------
    beta_history : np.ndarray
        Array of shape (n_patients, n_iterations) with the beta estimate for each patient.
    assignments : np.ndarray
        Array of shape (n_patients,) containing the (final) subtype assignment for each patient.
    n_subtypes : int
        Number of subtypes.
    beta_true : np.ndarray
        Array of shape (n_patients,) with true beta values.
    subtype_labels : Optional[Sequence[str]]
        Custom labels for subtypes; defaults to "Subtype {i}".
    subtype_mapping : Optional[np.ndarray]
        Optional mapping array where mapping[fitted_subtype] = true_subtype.
        If provided, labels will show "Fitted {i} -> True {mapping[i]}".
    """
    if beta_history.ndim != 2:
        raise ValueError(f"beta_history is expected to be 2-D (patients x iterations); got shape {beta_history.shape}")

    n_patients, n_iterations = beta_history.shape
    if assignments.shape[0] != n_patients:
        raise ValueError("assignments must have the same length as beta_history rows.")
    if beta_true.shape[0] != n_patients:
        raise ValueError("beta_true must have the same length as beta_history rows.")

    iteration_idx = np.arange(n_iterations)
    palette = sns.color_palette("tab10", n_subtypes)
    labels = [f"Subtype {i}" for i in range(n_subtypes)] if subtype_labels is None else list(subtype_labels)

    # Create figure with error plot and two histograms
    fig, (ax_error, ax_hist_all, ax_hist_subtype) = plt.subplots(
        1, 3, figsize=(18, 5)
    )

    # Compute errors for each subtype
    for subtype in range(n_subtypes):
        mask = assignments == subtype
        if not np.any(mask):
            continue

        subtype_history = beta_history[mask]  # (n_patients_subtype, n_iterations)
        true_values = beta_true[mask][:, None]  # (n_patients_subtype, 1)
        
        # Compute absolute error |beta_pred - beta_true| for each iteration
        abs_errors = np.abs(subtype_history - true_values)  # (n_patients_subtype, n_iterations)
        
        # Mean and std of absolute errors across patients for each iteration
        mean_abs_errors = abs_errors.mean(axis=0)  # (n_iterations,)
        std_abs_errors = abs_errors.std(axis=0)   # (n_iterations,)
        
        color = palette[subtype % len(palette)]
        label = labels[subtype] if subtype < len(labels) else f"Subtype {subtype}"

        # Plot mean absolute error with std bands
        ax_error.plot(iteration_idx, mean_abs_errors, color=color, label=label, linewidth=2)
        ax_error.fill_between(iteration_idx, mean_abs_errors - std_abs_errors, mean_abs_errors + std_abs_errors, 
                             alpha=0.2, color=color)

    ax_error.set_title("Mean absolute error (|beta_pred - beta_true|) by subtype")
    ax_error.set_ylabel("Mean absolute error")
    ax_error.set_xlabel("iteration")
    ax_error.grid(True, alpha=0.3)
    ax_error.legend()

    # Histogram: final beta vs true beta (overall)
    beta_final = beta_history[:, -1]
    
    # Create histogram comparing final and true beta
    bins = np.linspace(min(np.min(beta_true), np.min(beta_final)), 
                      max(np.max(beta_true), np.max(beta_final)), 30)
    
    ax_hist_all.hist(beta_true, bins=bins, alpha=0.6, label='True beta', color='gray', edgecolor='black')
    ax_hist_all.hist(beta_final, bins=bins, alpha=0.6, label='Final beta', color='steelblue', edgecolor='black')
    ax_hist_all.set_title("Distribution: Final beta vs True beta")
    ax_hist_all.set_xlabel("beta")
    ax_hist_all.set_ylabel("count")
    ax_hist_all.legend()
    ax_hist_all.grid(True, alpha=0.3)

    # Histogram: final beta by subtype
    for subtype in range(n_subtypes):
        mask = assignments == subtype
        if not np.any(mask):
            continue
        
        beta_final_subtype = beta_history[mask, -1]
        color = palette[subtype % len(palette)]
        label = labels[subtype] if subtype < len(labels) else f"Subtype {subtype}"
        
        ax_hist_subtype.hist(beta_final_subtype, bins=bins, alpha=0.6, label=label, 
                            color=color, edgecolor='black')
    
    ax_hist_subtype.set_title("Final beta distribution by subtype")
    ax_hist_subtype.set_xlabel("beta")
    ax_hist_subtype.set_ylabel("count")
    ax_hist_subtype.legend()
    ax_hist_subtype.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_theta_history_by_subtype(
    cluster_f_history: np.ndarray,
    cluster_scalar_K_history: np.ndarray,
    s_history: np.ndarray,
    n_subtypes: int,
    n_biomarkers: int,
    f_true_list: Sequence[np.ndarray],
    scalar_K_true_list: Sequence[float],
    s_true: np.ndarray,
    subtype_labels: Optional[Sequence[str]] = None,
    subtype_mapping: Optional[np.ndarray] = None,
) -> None:
    """
    Plot mean absolute error for theta parameters (f, scalar_K, s) over iterations by subtype.
    
    Parameters
    ----------
    cluster_f_history : np.ndarray
        Array of shape (n_subtypes, n_biomarkers, n_iterations) with f history for each subtype.
    cluster_scalar_K_history : np.ndarray
        Array of shape (n_subtypes, n_iterations) with scalar_K history for each subtype.
    s_history : np.ndarray
        Array of shape (n_biomarkers, n_iterations) with global s history.
    n_subtypes : int
        Number of subtypes.
    n_biomarkers : int
        Number of biomarkers.
    f_true_list : Sequence[np.ndarray]
        List of true f arrays, one per subtype. Each should be shape (n_biomarkers,).
    scalar_K_true_list : Sequence[float]
        List of true scalar_K values, one per subtype.
    s_true : np.ndarray
        True global s array, shape (n_biomarkers,).
    subtype_labels : Optional[Sequence[str]]
        Custom labels for subtypes; defaults to "Subtype {i}".
    subtype_mapping : Optional[np.ndarray]
        Optional mapping array where mapping[fitted_subtype] = true_subtype.
        If provided, will map fitted subtypes to true subtypes for comparison.
    """
    # Validate shapes
    if cluster_f_history.shape != (n_subtypes, n_biomarkers, cluster_f_history.shape[2]):
        raise ValueError(f"cluster_f_history expected shape (n_subtypes, n_biomarkers, n_iterations), got {cluster_f_history.shape}")
    if cluster_scalar_K_history.shape != (n_subtypes, cluster_f_history.shape[2]):
        raise ValueError(f"cluster_scalar_K_history expected shape (n_subtypes, n_iterations), got {cluster_scalar_K_history.shape}")
    if s_history.shape != (n_biomarkers, cluster_f_history.shape[2]):
        raise ValueError(f"s_history expected shape (n_biomarkers, n_iterations), got {s_history.shape}")
    if len(f_true_list) != n_subtypes:
        raise ValueError(f"f_true_list must have length {n_subtypes}, got {len(f_true_list)}")
    if len(scalar_K_true_list) != n_subtypes:
        raise ValueError(f"scalar_K_true_list must have length {n_subtypes}, got {len(scalar_K_true_list)}")
    if s_true.shape != (n_biomarkers,):
        raise ValueError(f"s_true expected shape (n_biomarkers,), got {s_true.shape}")
    
    n_iterations = cluster_f_history.shape[2]
    iteration_idx = np.arange(n_iterations)
    palette = sns.color_palette("tab10", n_subtypes)
    
    # Use subtype_mapping if provided to create better labels and map to true subtypes
    if subtype_labels is None:
        if subtype_mapping is not None:
            labels = [f"Fitted {i} -> True {subtype_mapping[i]}" for i in range(n_subtypes)]
        else:
            labels = [f"Subtype {i}" for i in range(n_subtypes)]
    else:
        labels = list(subtype_labels)
    
    # Create figure with three subplots for f, scalar_K, and s
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_f, ax_scalar_K, ax_s = axes
    
    # Plot f error by subtype
    for subtype in range(n_subtypes):
        f_history = cluster_f_history[subtype]  # (n_biomarkers, n_iterations)
        # Map to true subtype if mapping provided
        if subtype_mapping is not None:
            true_subtype_idx = subtype_mapping[subtype]
            f_true = np.asarray(f_true_list[true_subtype_idx])
        else:
            f_true = np.asarray(f_true_list[subtype])
        
        # Compute absolute error |f_pred - f_true| for each biomarker at each iteration
        f_errors = np.abs(f_history - f_true[:, None])  # (n_biomarkers, n_iterations)
        
        # Mean and std across biomarkers for each iteration
        mean_f_errors = f_errors.mean(axis=0)  # (n_iterations,)
        std_f_errors = f_errors.std(axis=0)   # (n_iterations,)
        
        color = palette[subtype % len(palette)]
        label = labels[subtype] if subtype < len(labels) else f"Subtype {subtype}"
        
        ax_f.plot(iteration_idx, mean_f_errors, color=color, label=label, linewidth=2)
        ax_f.fill_between(iteration_idx, mean_f_errors - std_f_errors, mean_f_errors + std_f_errors,
                         alpha=0.2, color=color)
    
    ax_f.set_title("Mean absolute error: f by subtype")
    ax_f.set_ylabel("Mean |f_pred - f_true|")
    ax_f.set_xlabel("iteration")
    ax_f.grid(True, alpha=0.3)
    ax_f.legend()
    
    # Plot scalar_K error by subtype
    for subtype in range(n_subtypes):
        scalar_K_history = cluster_scalar_K_history[subtype]  # (n_iterations,)
        # Map to true subtype if mapping provided
        if subtype_mapping is not None:
            true_subtype_idx = subtype_mapping[subtype]
            scalar_K_true = scalar_K_true_list[true_subtype_idx]
        else:
            scalar_K_true = scalar_K_true_list[subtype]
        
        # Compute absolute error |scalar_K_pred - scalar_K_true|
        scalar_K_errors = np.abs(scalar_K_history - scalar_K_true)  # (n_iterations,)
        
        color = palette[subtype % len(palette)]
        label = labels[subtype] if subtype < len(labels) else f"Subtype {subtype}"
        
        ax_scalar_K.plot(iteration_idx, scalar_K_errors, color=color, label=label, linewidth=2)
    
    ax_scalar_K.set_title("Absolute error: scalar_K by subtype")
    ax_scalar_K.set_ylabel("|scalar_K_pred - scalar_K_true|")
    ax_scalar_K.set_xlabel("iteration")
    ax_scalar_K.grid(True, alpha=0.3)
    ax_scalar_K.legend()
    
    # Plot s error (global, not per subtype)
    s_errors = np.abs(s_history - s_true[:, None])  # (n_biomarkers, n_iterations)
    mean_s_errors = s_errors.mean(axis=0)  # (n_iterations,)
    std_s_errors = s_errors.std(axis=0)   # (n_iterations,)
    
    ax_s.plot(iteration_idx, mean_s_errors, color='black', label='Global s', linewidth=2)
    ax_s.fill_between(iteration_idx, mean_s_errors - std_s_errors, mean_s_errors + std_s_errors,
                     alpha=0.2, color='black')
    
    ax_s.set_title("Mean absolute error: s (global)")
    ax_s.set_ylabel("Mean |s_pred - s_true|")
    ax_s.set_xlabel("iteration")
    ax_s.grid(True, alpha=0.3)
    ax_s.legend()
    
    plt.tight_layout()
    plt.show()


def plot_assignment_history(assignment_history: np.ndarray, max_patients: int = 10) -> None:
    """
    Visualize how cluster assignments evolve for a subset of patients.

    Parameters
    ----------
    assignment_history : np.ndarray
        Array of shape (n_patients, n_iterations) tracking subtype assignments.
    max_patients : int
        Maximum number of patients to display (rows). Patients are shown in index order.
    """
    if assignment_history.ndim != 2:
        raise ValueError("assignment_history must be a 2-D array (patients x iterations).")

    n_patients, n_iterations = assignment_history.shape
    num_rows = min(max_patients, n_patients)
    subset = assignment_history[:num_rows]

    plt.figure(figsize=(max(6, 0.6 * n_iterations), max(3, 0.4 * num_rows)))
    sns.heatmap(
        subset,
        cmap="tab20",
        cbar=True,
        linewidths=0.5,
        linecolor="white",
        square=False,
        xticklabels=np.arange(n_iterations),
        yticklabels=[f"Patient {i}" for i in range(num_rows)],
    )
    plt.title("Assignment history (patient vs iteration)")
    plt.xlabel("iteration")
    plt.ylabel("patient")
    plt.tight_layout()
    plt.show()


def plot_assignment_stability(assignment_history: np.ndarray) -> None:
    """
    Plot the fraction of patients whose assignments change at each iteration.
    """
    if assignment_history.ndim != 2 or assignment_history.shape[1] < 2:
        raise ValueError("assignment_history must be 2-D with at least two iterations recorded.")

    changes = assignment_history[:, 1:] != assignment_history[:, :-1]
    fraction_changed = changes.mean(axis=0)
    iteration_idx = np.arange(1, assignment_history.shape[1])

    plt.figure(figsize=(8, 3.5))
    plt.plot(iteration_idx, fraction_changed, marker="o", color="#DD8452")
    plt.title("Assignment stability across EM iterations")
    plt.xlabel("iteration")
    plt.ylabel("fraction of patients that changed subtype")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_beta_comparison(
    beta_true: np.ndarray,
    beta_init: np.ndarray,
    beta_final: np.ndarray,
    title: str = "Beta comparison",
) -> None:
    """
    Scatter plots comparing initial and final beta estimates against ground truth.
    """
    if not (len(beta_true) == len(beta_init) == len(beta_final)):
        raise ValueError("beta_true, beta_init, and beta_final must have the same length.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    limits = [
        min(np.min(beta_true), np.min(beta_init), np.min(beta_final)),
        max(np.max(beta_true), np.max(beta_init), np.max(beta_final)),
    ]

    axes[0].scatter(beta_true, beta_init, alpha=0.6, color="#55A868", edgecolors="white")
    axes[0].plot(limits, limits, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Initial vs true beta")
    axes[0].set_xlabel("True beta")
    axes[0].set_ylabel("Estimated beta")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(beta_true, beta_final, alpha=0.6, color="#C44E52", edgecolors="white")
    axes[1].plot(limits, limits, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Final vs true beta")
    axes[1].set_xlabel("True beta")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_true_vs_predicted_subtype_trajectories(
    n_subtypes: int,
    f_true_list: Sequence[np.ndarray],
    scalar_K_true_list: Sequence[float],
    f_pred_list: Sequence[np.ndarray],
    scalar_K_pred_list: Sequence[float],
    K: np.ndarray,
    t_span: np.ndarray,
    n_biomarkers: int,
    solve_system_fn,
    subtype_mapping: Optional[np.ndarray] = None,
) -> None:
    """
    Overlay true vs predicted biomarker trajectories for each fitted subtype.

    One panel per fitted subtype. When subtype_mapping is provided, each panel
    shows the fitted trajectory vs the true trajectory of the mapped-to true subtype
    (e.g. "Fitted 3 -> True 0" when there are more fitted than true subtypes).

    Parameters
    ----------
    subtype_mapping : Optional[np.ndarray]
        Optional mapping array where mapping[fitted_subtype] = true_subtype.
        If provided, number of panels = len(f_pred_list) and each panel compares
        fitted trajectory to the mapped true subtype.
    """
    n_fitted = len(f_pred_list)
    if n_fitted == 0:
        raise ValueError("f_pred_list must not be empty.")
    if len(scalar_K_pred_list) != n_fitted:
        raise ValueError(
            f"scalar_K_pred_list must have length {n_fitted} (number of fitted subtypes), got {len(scalar_K_pred_list)}."
        )
    if subtype_mapping is not None:
        if len(subtype_mapping) != n_fitted:
            raise ValueError(
                f"subtype_mapping length must match number of fitted subtypes ({n_fitted}), got {len(subtype_mapping)}."
            )
        max_true_idx = int(np.max(subtype_mapping))
        if len(f_true_list) <= max_true_idx or len(scalar_K_true_list) <= max_true_idx:
            raise ValueError(
                f"f_true_list and scalar_K_true_list must have at least {max_true_idx + 1} elements for subtype_mapping."
            )
    else:
        if len(f_true_list) < n_subtypes or len(f_true_list) < n_fitted:
            raise ValueError("f_true_list must have at least n_subtypes (or n_fitted) elements when subtype_mapping is None.")
        if len(scalar_K_true_list) < n_fitted:
            raise ValueError("scalar_K_true_list must have at least n_fitted elements when subtype_mapping is None.")
        if n_fitted != n_subtypes:
            raise ValueError("When subtype_mapping is None, len(f_pred_list) must equal n_subtypes.")

    colors = sns.color_palette("deep", n_biomarkers)
    fig, axes = plt.subplots(n_fitted, 1, figsize=(10, max(3 * n_fitted, 4)), sharex=True)
    if n_fitted == 1:
        axes = [axes]

    for subtype in range(n_fitted):
        ax = axes[subtype]
        # Map to true subtype if mapping provided
        if subtype_mapping is not None:
            true_subtype_idx = int(subtype_mapping[subtype])
            f_true = np.asarray(f_true_list[true_subtype_idx])
            scalar_true = float(scalar_K_true_list[true_subtype_idx])
        else:
            f_true = np.asarray(f_true_list[subtype])
            scalar_true = float(scalar_K_true_list[subtype])

        f_pred = np.asarray(f_pred_list[subtype])
        scalar_pred = float(scalar_K_pred_list[subtype])

        traj_true = solve_system_fn(np.zeros(n_biomarkers), f_true, K, t_span, scalar_true)
        traj_pred = solve_system_fn(np.zeros(n_biomarkers), f_pred, K, t_span, scalar_pred)

        for biom_idx in range(n_biomarkers):
            color = colors[biom_idx % len(colors)]
            label_true = f"Biomarker {biom_idx + 1} (true)" if subtype == 0 else None
            label_pred = f"Biomarker {biom_idx + 1} (pred)" if subtype == 0 else None
            ax.plot(t_span, traj_true[biom_idx], color=color, linestyle="-", alpha=0.8, label=label_true)
            ax.plot(t_span, traj_pred[biom_idx], color=color, linestyle="--", alpha=0.8, label=label_pred)

        if subtype_mapping is not None:
            true_subtype_idx = int(subtype_mapping[subtype])
            ax.set_title(f"Fitted {subtype} -> True {true_subtype_idx}: trajectories")
        else:
            ax.set_title(f"Subtype {subtype}: true vs predicted trajectories")
        ax.set_ylabel("trajectory value")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()
    
def plot_lse(lse_array: np.ndarray) -> None:
    """
    Plots LSE trace across iterations.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(lse_array)), lse_array)
    plt.title("Least squares error (LSE) per iterations")
    plt.yscale('log')
    plt.xlabel("iteration")
    plt.ylabel("LSE")
    plt.show()

def plot_cog_regression_history(cog_history: np.ndarray, labels: list):
    n_params, num_iterations = cog_history.shape
    plt.figure(figsize=(8, 5))
    for i in range(n_params):
        label = f"{labels[i]}_coeff" if i < n_params - 1 else "b"
        plt.plot(range(num_iterations), cog_history[i,:], label=label)
    plt.legend
    plt.xlabel("iteration")
    plt.ylabel("estimated coefficient value")
    plt.title("cog parameters")
    plt.grid(True)
    plt.show()



def plot_all_patient_regression_lines_grid_nhy(X, dt, ids, beta, t_span, nhy, model=None,
                                               biomarker_indices=None, biomarker_labels=None,
                                               max_lines=500, t_max=40):
    """
    Plot regression lines per patient for each selected biomarker on a grid of subplots.
    Each line is color-coded by the patient's mean NHY score.
    """
    if biomarker_indices is None:
        biomarker_indices = list(range(X.shape[1]))

    unique_ids = np.unique(ids)
    pid_to_index = {pid: idx for idx, pid in enumerate(unique_ids)}

    # compute mean NHY score per patient
    mean_nhy = {}
    for pid in unique_ids:
        nhy_i = nhy[ids == pid]
        mean_nhy[pid] = np.mean(nhy_i) if len(nhy_i) > 0 else np.nan

    # colormap setup
    cmap = cm.plasma  # viridis # inferno
    nhy_vals = np.array(list(mean_nhy.values()))
    norm = colors.Normalize(vmin=np.nanmin(nhy_vals), vmax=np.nanmax(nhy_vals))

    n_plots = len(biomarker_indices)
    n_cols = 5
    n_rows = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=True)
    axes = axes.flatten()

    for plot_idx, j in enumerate(biomarker_indices):
        ax = axes[plot_idx]
        title = biomarker_labels[j] if biomarker_labels else f"biomarker {j}"
        ax.set_title(f"{title}")
        
        line_count = 0
        for pid in unique_ids:
            if line_count >= max_lines:
                break
            mask = (ids == pid)
            if np.sum(mask) < 2:
                continue

            X_i = X[mask, j]
            dt_i = dt[mask]
            beta_i = beta[pid_to_index[pid]]
            t_ij = dt_i + beta_i

            nhy_mean = mean_nhy[pid]
            if np.isnan(nhy_mean):
                continue
            line_color = cmap(norm(nhy_mean))

            model_i = LinearRegression().fit(t_ij.reshape(-1, 1), X_i)
            t_fit = np.linspace(t_ij.min(), t_ij.max(), 20)
            x_fit = model_i.predict(t_fit.reshape(-1, 1))

            ax.plot(t_fit, x_fit, color=line_color, alpha=0.8, linewidth=1)
            line_count += 1

        # plot model-predicted trajectory
        if model is not None:
            ax.plot(t_span, model[j], color='black', linewidth=2, label="Model")
            ax.legend()
            
        # determine subplot grid position
        row_idx = plot_idx // n_cols
        col_idx = plot_idx % n_cols

        # only show x-labels on bottom row
        if row_idx == n_rows - 1:
            ax.set_xlabel("time (y)")
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        # only show y-labels on leftmost column
        if col_idx == 0:
            ax.set_ylabel("biomarker value")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.set_xlim(0, t_max)

    # remove unused axes
    for k in range(n_plots, len(axes)):
        fig.delaxes(axes[k])

    # Add colorbar for NHY scale
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.88)  # make space for colorbar on right
    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Mean HY Score")
    
    plt.tight_layout()
    plt.show()

def plot_violin_nhy_vs_beta(ids, dt, nhy, beta):
    unique_ids = np.unique(ids)
    pair = []

    pid_to_beta = {pid: beta[i] for i, pid in enumerate(unique_ids)}

    for pid in unique_ids:
        mask = (ids == pid)
        dt_i = dt[mask]
        nhy_i = nhy[mask]
        idx_min_dt = np.argmin(dt_i)
        nhy_first = nhy_i[idx_min_dt]
        beta_i = pid_to_beta[pid]

        pair.append({"NHY": nhy_first, "beta": beta_i})

    df = pd.DataFrame(pair)

    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x="NHY", y="beta", palette="plasma", inner="box")
    plt.title("initial HY vs. beta")
    plt.xlabel("initial HY")
    plt.ylabel("beta")
    plt.show()
    
def plot_violin_nhy_vs_tij(dt, ids, beta, nhy):
    unique_ids = np.unique(ids)
    pid_to_index = {pid: idx for idx, pid in enumerate(unique_ids)}
    t_ij = np.array([dt_i + beta[pid_to_index[pid]] for dt_i, pid in zip(dt, ids)])

    df = pd.DataFrame({
        "t_ij": t_ij,
        "NHY": nhy
    })

    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x="NHY", y="t_ij", palette="plasma", inner="box")
    plt.xlabel("HY stage")
    plt.ylabel("t_ij")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_patient_trajectories_by_biomarker(X, biomarker_indices):
    """
    For each biomarker in `biomarker_indices`, create a subplot showing that biomarker's
    trajectory for all patients. Lines are color-coded by `subtype_true` using tab10.
    Assumes each patient dict has:
      - "X_pred_subject": array of shape (n_biomarkers, T)
      - "subtype_true": int in [0..9] (tab10)
    """
    # Ensure X is iterable as a list of dicts
    if isinstance(X, np.ndarray):
        X_list = list(X)
    else:
        X_list = X

    if len(X_list) == 0:
        raise ValueError("X is empty.")

    # infer time axis from the first patient
    xps = X_list[0]["X_pred_subject"]
    # xps expected shape: (n_biomarkers, T)
    if xps.ndim != 2:
        raise ValueError(f'X_pred_subject expected 2D (n_biomarkers, T), got shape {xps.shape}')
    T = xps.shape[1]
    # If you stored a real t_span, replace the line below with: t = X_list[0]["t_span"]
    t = np.linspace(0.0, 1.0, T)  # normalized time if t_span not available

    # figure layout: <=5 per row
    n_plots = len(biomarker_indices)
    n_cols = min(5, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 4.5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes[:, None]

    plot_idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if plot_idx >= n_plots:
                # hide any extra axes
                ax.axis('off')
                continue

            b = biomarker_indices[plot_idx]

            # draw each patient's trajectory for biomarker b
            # color by subtype_true
            for i in range(len(X_list)):
                p = X_list[i]
                subtype = int(p.get("subtype_true", 0))
                color = cmap(subtype % cmap.N)

                xps_i = p["X_pred_subject"]
                if xps_i.ndim != 2 or b >= xps_i.shape[0]:
                    raise ValueError(f'Patient {i} has invalid X_pred_subject shape {xps_i.shape} or biomarker index {b}')
                y = xps_i[b]  # shape (T,)
                ax.plot(t, y, color=color, linewidth=1.2, alpha=0.9)

            ax.set_title(f"biomarker {b+1}")
            ax.set_xlabel("time (y)")
            ax.set_ylabel("biomarker progression")
            ax.set_aspect('equal', adjustable='box')  # square plot

            plot_idx += 1

    # Build a simple legend for subtypes (0..max present)
    # Find unique subtypes present in X
    subtypes_present = sorted({int(p.get("subtype_true", 0)) for p in X_list})
    legend_lines = []
    legend_labels = []
    for s in subtypes_present:
        legend_lines.append(plt.Line2D([0], [0], color=cmap(s % cmap.N), lw=3))
        legend_labels.append(f"Subtype {s}")

    # Put legend under the grid
    fig.legend(legend_lines, legend_labels, loc="lower center", ncol=min(5, len(legend_labels)), frameon=False)
    plt.subplots_adjust(bottom=0.12, wspace=0.3, hspace=0.35)
    plt.show()


def plot_assignment_accuracy_history(
    assignment_history: np.ndarray,
    true_assignments: np.ndarray,
    subtype_mapping: Optional[np.ndarray] = None,
    title: str = "Assignment Accuracy Over Iterations",
) -> None:
    """
    Plot the percentage of correct cluster assignments over EM iterations.
    
    Parameters
    ----------
    assignment_history : np.ndarray
        Array of shape (n_patients, n_iterations) tracking subtype assignments over iterations.
    true_assignments : np.ndarray
        Array of shape (n_patients,) with true subtype assignments.
    subtype_mapping : Optional[np.ndarray]
        Optional mapping array where mapping[fitted_subtype] = true_subtype.
        If provided, will map fitted assignments to true subtypes before comparison.
    title : str
        Title for the plot.
    """
    if assignment_history.ndim != 2:
        raise ValueError(f"assignment_history must be 2-D (patients x iterations), got shape {assignment_history.shape}")
    
    n_patients, n_iterations = assignment_history.shape
    if len(true_assignments) != n_patients:
        raise ValueError(f"true_assignments must have length {n_patients}, got {len(true_assignments)}")
    
    # Compute accuracy for each iteration
    accuracy_history = []
    for iter_idx in range(n_iterations):
        assignments_iter = assignment_history[:, iter_idx]
        
        # Apply subtype mapping if provided
        if subtype_mapping is not None:
            # Map fitted assignments to true subtypes
            mapped_assignments = np.array([subtype_mapping[a] for a in assignments_iter])
        else:
            mapped_assignments = assignments_iter
        
        # Compute accuracy (percentage of correct assignments)
        correct = np.sum(mapped_assignments == true_assignments)
        accuracy = 100.0 * correct / n_patients
        accuracy_history.append(accuracy)
    
    accuracy_history = np.array(accuracy_history)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(accuracy_history, 'b-', linewidth=2, marker='o', markersize=4)
    plt.axhline(y=100.0, color='green', linestyle='--', alpha=0.5, label='Perfect (100%)')
    plt.xlabel('Iteration')
    plt.ylabel('Assignment Accuracy (%)')
    plt.title(title)
    plt.ylim([0, 105])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

