import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns

# TODO: add general params for all plots

def plot_biomarker_trajectories(biom_trajectories: np.ndarray, t_span: np.ndarray, n_biomarkers: int = 68) -> None:
    """
    Plots ground truth biomarker trajectories over time.
    """
    plt.figure(figsize=(15,6))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_biomarkers))
    for b in range(n_biomarkers):
        plt.plot(t_span, biom_trajectories[b], color = colors[b])
    plt.title("biomarker trajectories")
    plt.legend()
    plt.show()

def plot_true_observations(df: pd.DataFrame, t: np.ndarray, x_true: np.ndarray, patient_idx=None) -> None:
    """
    Overlays observed biomarker values from selected patients on the true model trajectories.
    Each patient gets a unique color and marker.
    """
    if patient_idx is None:
        patient_idx = [0, 1, 2, 3, 4]

    n_biomarkers = x_true.shape[0]
    colors = plt.get_cmap("tab10").colors  # ehhh should be more than 10
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*', '<', '>'] # incase I want more

    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot true curves (gray background)
    for i in range(n_biomarkers):
        ax.plot(t, x_true[i], color="k", alpha=0.5, linewidth=1)

    # Plot observations
    for color_index, patient in enumerate(patient_idx):
        patient_data = df[df["patient_id"] == patient]
        t_ij = patient_data["beta_true"] + patient_data["dt"]
        marker = markers[color_index % len(markers)]
        color = colors[color_index % len(colors)]

        for i in range(n_biomarkers):
            y = patient_data[f"biomarker_{i+1}"]
            ax.scatter(t_ij, y, color=color, marker=marker, s=30, label=f"Patient {patient}" if i == 0 else None)

    ax.set_title("Synthetic patient observations timepoints on groundtruth biomarker trajectories")
    ax.set_xlabel("Time")
    ax.set_ylabel("Biomarker Value")
    ax.legend(ncol=len(patient_idx), fontsize=9)
    # plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    
def plot_lse(lse_array: np.ndarray) -> None:
    """
    Plots LSE trace across iterations.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(lse_array)), lse_array)
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
        mean_nhy[pid] = np.max(nhy_i) if len(nhy_i) > 0 else np.nan
        # mean_nhy[pid] = np.mean(nhy_i) if len(nhy_i) > 0 else np.nan

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
    cbar.set_label("final HY Score")
    
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