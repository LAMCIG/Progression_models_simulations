import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment
import statsmodels.formula.api as smf
from typing import Sequence

def solve_system(x0: np.ndarray, f: np.ndarray, K: np.ndarray, t_span: np.ndarray, scalar_K: float = 1.0) -> np.ndarray: #, alpha: float = 1.0) -> np.ndarray:
    """
    Solves the multivariate logistic ODE system given initial conditions and parameters.

    Parameters
    ----------
    x0 : np.ndarray
        Initial state vector (typically zeros).
    f : np.ndarray
        Forcing term vector.
    K : np.ndarray
        Connectivity matrix.
    t_span : np.ndarray
        Array of time points to solve over.

    Returns
    -------
    np.ndarray
        Simulated biomarker trajectories of shape (n_biomarkers, len(t_span)).
    """
    # eps = 1e-2
    
    def ode_system(t, x):
        # x = np.clip(x, 0.0 + eps, 1.0 - eps)  # prevent overshoot near upper bound
        dxdt =  (np.eye(K.shape[0]) - np.diag(x)) @ ((scalar_K * K @ x) + f) # TODO: double check position of f wrt parenthesis
        return dxdt
    # def jacobian_ode(t, x):
    #     J = (1 - x)[:, None] * K
    #     J[np.diag_indices_from(J)] = -((K @ x) + f)
    #     return J
    
    # check parent function declaration for attributes
    # def jacobian_ode(t, x, alpha = alpha) -> np.ndarray: # returns matrix
    #     J = (1 - x)[:, None] * (alpha * K) # for non-diag
    #     Kx_plus_f = alpha * (K @ x) + f # diagonal entries
    #     J[np.diag_indices_from(J)] = -Kx_plus_f
    #     return J
    
    def jacobian_ode(t, x):
        J = (1 - x)[:, None] * (scalar_K * K)
        J[np.diag_indices_from(J)] = -(scalar_K * (K @ x) + f)
        return J    

    sol = solve_ivp(
        ode_system,
        [t_span[0], t_span[-1]],
        x0,
        t_eval=t_span,
        method="LSODA",
        jac=jacobian_ode,
        # rtol=1e-8,#1e-6,
        # atol=1e-10,#1e-8
        # reminder: LSODA ignores min and max step
    )


    return sol.y

def initialize_beta(ids: np.ndarray, beta_range: tuple = (0, 12), rng: np.random.Generator = None) -> np.ndarray:
    """
    Uniformly randomly initialize beta values for each unique patient ID.

    Returns
    -------
    np.ndarray
        1D array of beta values indexed by patient.
    """
    if rng is None:
        rng = np.random.default_rng(75)
    patient_ids = np.unique(ids)
    initial_beta = rng.uniform(beta_range[0], beta_range[1], size=len(patient_ids))
    
    return initial_beta

def initialize_f_eigen(K: np.ndarray, jitter_strength: float = 0.05, n_eigs: int = 1, rng: np.random.RandomState = np.random.RandomState(75)) -> np.ndarray:
    """
    Initialize forcing term f using top eigenvectors of the connectivity matrix.
    """
    w, V = np.linalg.eigh(K)  # returns (eigenvalues, eigenvectors)
    order = np.argsort(np.abs(w))[::-1]   # sort by eigenvalue, descending?
    V = V[:, order]

    # Take top n_eigs
    f_list = []
    for i in range(min(n_eigs, V.shape[1])):
        f = np.abs(V[:, i])
        # f /= f.mean()        # scale so mean ~1
        # f *= 0.05           
        if rng is not None:
            f += rng.normal(0, jitter_strength, size=f.shape)  # apply jitter
            f = np.clip(f, 0.0, None)
        f_list.append(f)

    return np.vstack(f_list)

def set_diagonal_K(K: np.ndarray, s: float = 1.0, k: float = 0.05): # TODO: ask BG if K should be zeroed beforehand?
    """
    Params:
    s: global strength of connections? scales everything
    k: 
    
    """
    K = np.array(K, dtype=float, copy=True)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K needs to be a square matrix, not {K.shape}")
    
    n = K.shape[0]
    I = np.eye(n)
    
    K_diag = s * K + k * I
    return K_diag
    # then return K with a diag
    
def ensure_2d_cog(c, n_rows_expected: int) -> np.ndarray:
    c = np.asarray(c)
    # (n_obs,) -> (n_obs, 1)
    if c.ndim == 1:
        c = c.reshape(-1, 1)

    # (1, n_obs) -> (n_obs, 1)
    if c.ndim == 2 and c.shape[0] == 1 and c.shape[1] == n_rows_expected:
        c = c.T

    # Final check: first dim must match number of observations for this patient
    if c.ndim != 2 or c.shape[0] != n_rows_expected:
        raise ValueError(
            f"cog shape mismatch; expected first dim {n_rows_expected}, got {c.shape}"
        )
    return c


def get_subtype_mapping_from_f(
    fitted_f_list: Sequence[np.ndarray],
    true_f_list: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Create a mapping from fitted subtype indices to true subtype indices based on f parameters.
    
    Uses Hungarian algorithm to find the best match between fitted and true subtypes
    based on Euclidean distance between f vectors. Returns an array where index = fitted subtype, 
    value = corresponding true subtype.
    
    Parameters
    ----------
    fitted_f_list : Sequence[np.ndarray]
        List of fitted f arrays, one per subtype. Each should be shape (n_biomarkers,).
    true_f_list : Sequence[np.ndarray]
        List of true f arrays, one per subtype. Each should be shape (n_biomarkers,).
    
    Returns
    -------
    np.ndarray
        Mapping array where mapping[fitted_subtype] = true_subtype
        Shape: (n_subtypes,)
    """
    fitted_f_list = [np.asarray(f) for f in fitted_f_list]
    true_f_list = [np.asarray(f) for f in true_f_list]
    
    n_fitted = len(fitted_f_list)
    n_true = len(true_f_list)
    n_subtypes = max(n_fitted, n_true)
    
    # Build cost matrix: cost[i, j] = Euclidean distance between fitted_f[i] and true_f[j]
    cost_matrix = np.zeros((n_subtypes, n_subtypes))
    for i in range(n_fitted):
        for j in range(n_true):
            # Compute Euclidean distance between f vectors
            cost_matrix[i, j] = np.linalg.norm(fitted_f_list[i] - true_f_list[j])
    
    # Use Hungarian algorithm to find optimal assignment (minimize total distance)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create mapping: mapping[fitted_subtype] = true_subtype
    mapping = np.zeros(n_subtypes, dtype=int)
    for fitted_idx, true_idx in zip(row_ind, col_ind):
        mapping[fitted_idx] = true_idx
    
    return mapping


def get_subtype_mapping(fitted_assignments: np.ndarray, true_assignments: np.ndarray) -> np.ndarray:
    """
    Create a mapping from fitted subtype indices to true subtype indices.
    
    Uses Hungarian algorithm to find the best match between fitted and true subtypes
    based on patient overlap. Returns an array where index = fitted subtype, 
    value = corresponding true subtype.
    
    Parameters
    ----------
    fitted_assignments : np.ndarray
        Subtype assignments from fitted model (shape: n_patients,)
    true_assignments : np.ndarray
        True subtype assignments (shape: n_patients,)
    
    Returns
    -------
    np.ndarray
        Mapping array where mapping[fitted_subtype] = true_subtype
        Shape: (n_subtypes,)
    """
    fitted_assignments = np.asarray(fitted_assignments)
    true_assignments = np.asarray(true_assignments)
    
    if len(fitted_assignments) != len(true_assignments):
        raise ValueError(
            f"fitted_assignments and true_assignments must have same length, "
            f"got {len(fitted_assignments)} and {len(true_assignments)}"
        )
    
    n_fitted = len(np.unique(fitted_assignments))
    n_true = len(np.unique(true_assignments))
    n_subtypes = max(n_fitted, n_true)
    
    # Build confusion matrix: cost[i, j] = number of patients with fitted=i and true=j
    # We want to maximize agreement, so we use negative counts as costs
    cost_matrix = np.zeros((n_subtypes, n_subtypes))
    for i in range(n_subtypes):
        for j in range(n_subtypes):
            mask = (fitted_assignments == i) & (true_assignments == j)
            cost_matrix[i, j] = -np.sum(mask)  # Negative because we want to maximize
    
    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create mapping: mapping[fitted_subtype] = true_subtype
    mapping = np.zeros(n_subtypes, dtype=int)
    for fitted_idx, true_idx in zip(row_ind, col_ind):
        mapping[fitted_idx] = true_idx
    
    return mapping


def match_labels_best_overlap(em_labels: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """
    Match EM-assigned labels to true labels based on best overlap using Hungarian algorithm.
    
    This function finds the optimal permutation of EM labels that maximizes agreement
    with true labels. This is necessary because EM assigns arbitrary label numbers
    that don't necessarily correspond to true subtype indices.
    
    Parameters
    ----------
    em_labels : np.ndarray
        Labels assigned by EM algorithm (shape: n_patients,)
    true_labels : np.ndarray
        True subtype labels (shape: n_patients,)
    
    Returns
    -------
    np.ndarray
        Remapped EM labels that best match true labels (shape: n_patients,)
    """
    em_labels = np.asarray(em_labels)
    true_labels = np.asarray(true_labels)
    
    if len(em_labels) != len(true_labels):
        raise ValueError(f"em_labels and true_labels must have same length, got {len(em_labels)} and {len(true_labels)}")
    
    n_em_clusters = len(np.unique(em_labels))
    n_true_clusters = len(np.unique(true_labels))
    n_clusters = max(n_em_clusters, n_true_clusters)
    
    # Build confusion matrix: cost[i, j] = number of patients with em_label=i and true_label=j
    # We want to maximize agreement, so we use negative counts as costs
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            # Count how many patients have em_label=i and true_label=j
            mask = (em_labels == i) & (true_labels == j)
            cost_matrix[i, j] = -np.sum(mask)  # Negative because we want to maximize
    
    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create mapping from EM labels to true labels
    label_mapping = np.zeros(n_clusters, dtype=int)
    for em_idx, true_idx in zip(row_ind, col_ind):
        label_mapping[em_idx] = true_idx
    
    # Remap EM labels
    remapped_labels = label_mapping[em_labels]
    
    return remapped_labels


def print_parameter_comparison(
    fitted_f_list: Sequence[np.ndarray],
    fitted_scalar_K: float,
    fitted_s: np.ndarray,
    true_f_list: Sequence[np.ndarray],
    true_scalar_K_list: Sequence[float],
    true_s: np.ndarray,
    subtype_mapping: np.ndarray = None,
    n_subtypes: int = None,
):
    """
    Print a comparison of fitted vs true parameters (f, scalar_K, s).
    
    Parameters
    ----------
    fitted_f_list : Sequence[np.ndarray]
        List of fitted f arrays, one per subtype.
    fitted_scalar_K : float
        Fitted global scalar_K value (shared across all subtypes).
    fitted_s : np.ndarray
        Fitted global s array.
    true_f_list : Sequence[np.ndarray]
        List of true f arrays, one per subtype.
    true_scalar_K_list : Sequence[float]
        List of true scalar_K values, one per subtype (for comparison).
    true_s : np.ndarray
        True global s array.
    subtype_mapping : np.ndarray, optional
        Mapping array where mapping[fitted_subtype] = true_subtype.
        If None, assumes fitted subtype i corresponds to true subtype i.
    n_subtypes : int, optional
        Number of subtypes. If None, inferred from fitted_f_list.
    """
    if n_subtypes is None:
        n_subtypes = len(fitted_f_list)
    
    if subtype_mapping is None:
        subtype_mapping = np.arange(n_subtypes)
    
    
    # Compare f by subtype
    for fitted_subtype in range(n_subtypes):
        true_subtype = subtype_mapping[fitted_subtype]
        f_fitted = np.asarray(fitted_f_list[fitted_subtype])
        f_true = np.asarray(true_f_list[true_subtype])
        
        f_error = np.mean(np.abs(f_fitted - f_true))
        
        print(f"\nFitted Subtype {fitted_subtype} -> True Subtype {true_subtype}:")
        print(f"  f_fitted:      {f_fitted}")
        print(f"  f_true:        {f_true}")
    
    # Compare global scalar_K (single value, compare with average of true values)
    true_scalar_K_mean = np.mean(true_scalar_K_list)
    scalar_K_error = np.abs(fitted_scalar_K - true_scalar_K_mean)
    print(f"\nGlobal scalar_K:")
    print(f"  scalar_K_fitted: {fitted_scalar_K:.6f}")
    print(f"  scalar_K_true (mean): {true_scalar_K_mean:.6f}")
    if len(true_scalar_K_list) > 1:
        print(f"  scalar_K_true (per subtype): {true_scalar_K_list}")
    
    # Compare global s
    print(f"\nGlobal s:")
    print(f"  s_fitted:      {fitted_s}")
    print(f"  s_true:        {true_s}")


def _z(x):
    """Z-score normalization helper function."""
    x = np.asarray(x, float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    return (x - m) / (s if np.isfinite(s) and s > 0 else 1.0)


def build_severity_index(df: pd.DataFrame = None, cog: np.ndarray = None) -> np.ndarray:
    """
    Build a severity index from clinical scores.
    
    Works for both real data (with MCATOT, TD_score, PIGD_score) and synthetic data
    (with cognitive_score). For real data, higher severity = worse (lower MoCA, higher TD/PIGD).
    For synthetic data, uses cognitive_score directly.
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        DataFrame with clinical scores. Must have either:
        - For real data: "MCATOT", "TD_score", "PIGD_score" columns
        - For synthetic data: "cognitive_score" column
    cog : np.ndarray, optional
        Alternative input: array of cognitive scores (for synthetic data).
        If provided, df is ignored.
    
    Returns
    -------
    np.ndarray
        Severity index (higher = worse/later disease stage)
    """
    if cog is not None:
        # Synthetic data: use cognitive_score directly
        # For synthetic data, higher cognitive_score typically means later disease
        # We negate it to make higher = worse (consistent with real data)
        cog_array = np.asarray(cog, float)
        return -_z(cog_array)
    
    if df is None:
        raise ValueError("Either df or cog must be provided")
    
    # Check if this is real data (has MCATOT, TD_score, PIGD_score)
    if all(col in df.columns for col in ["MCATOT", "TD_score", "PIGD_score"]):
        moca = _z(df["MCATOT"].to_numpy())        # higher is better
        td = _z(df["TD_score"].to_numpy())
        pigd = _z(df["PIGD_score"].to_numpy())
        S = (-moca + td + pigd) / 3.0             # higher = worse / later
        return S
    # Check if this is synthetic data (has cognitive_score)
    elif "cognitive_score" in df.columns:
        cog_array = df["cognitive_score"].to_numpy()
        return -_z(cog_array)  # Negate to make higher = worse
    else:
        raise ValueError(
            "df must contain either (MCATOT, TD_score, PIGD_score) for real data "
            "or 'cognitive_score' for synthetic data"
        )


def fit_mixedlm_beta_from_clinical(df: pd.DataFrame = None, ids: np.ndarray = None, 
                                   dt: np.ndarray = None, t_max: float = 30,
                                   cog: np.ndarray = None, verbose: bool = False,
                                   rng: np.random.Generator = None) -> tuple:
    """
    Initialize beta values from clinical scores using mixed-effects linear model.
    
    Works for both real data (with MCATOT, TD_score, PIGD_score) and synthetic data
    (with cognitive_score). Fits a mixed-effects model: severity ~ dt + (1|id)
    and uses random intercepts to estimate patient-specific beta values.
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        DataFrame with clinical scores and time information.
        Must have columns: patient id, dt (or time), and clinical scores.
    ids : np.ndarray, optional
        Patient IDs (required if df not provided)
    dt : np.ndarray, optional
        Time since first visit (required if df not provided)
    t_max : float
        Maximum time value for clipping beta
    cog : np.ndarray, optional
        Cognitive scores array (for synthetic data, alternative to df)
    verbose : bool
        Whether to print summary statistics
    rng : np.random.Generator, optional
        Random number generator for jitter
    
    Returns
    -------
    tuple
        (initial_beta, pid_to_beta, result)
        initial_beta : np.ndarray of beta values indexed by unique patient IDs
        pid_to_beta : dict mapping patient ID to beta value
        result : fitted model result (or None if failed)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    # Build severity index
    if df is not None:
        S = build_severity_index(df=df)
        if ids is None:
            # Try to infer from df
            if "patient_id" in df.columns:
                ids = df["patient_id"].to_numpy()
            elif "subj_id" in df.columns:
                ids = df["subj_id"].to_numpy()
            else:
                raise ValueError("Cannot infer ids from df. Please provide ids parameter.")
        if dt is None:
            if "dt" in df.columns:
                dt = df["dt"].to_numpy()
            elif "time" in df.columns:
                dt = df["time"].to_numpy()
            else:
                raise ValueError("Cannot infer dt from df. Please provide dt parameter.")
    elif cog is not None and ids is not None and dt is not None:
        S = build_severity_index(cog=cog)
    else:
        raise ValueError("Must provide either (df) or (ids, dt, cog)")
    
    # Create DataFrame for mixed model
    dfm = pd.DataFrame({"id": ids, "dt": dt.astype(float), "S": S})
    
    # Keep longitudinal subjects (at least 2 observations)
    good_ids = dfm.groupby("id").size().pipe(lambda s: s[s >= 2]).index
    dfm = dfm[dfm["id"].isin(good_ids)].copy()
    
    # Clean + tiny jitter on dt to avoid exact duplicate rows
    dfm = dfm.replace([np.inf, -np.inf], np.nan).dropna(subset=["id", "dt", "S"])
    dfm["dt"] = dfm["dt"] + 1e-6 * rng.standard_normal(len(dfm))
    
    # Fit: severity ~ dt + (1|id)
    model = smf.mixedlm("S ~ dt", data=dfm, groups=dfm["id"], re_formula="1")
    result = None
    for meth in ("bfgs", "nm"):
        try:
            result = model.fit(method=meth, reml=True, maxiter=500, disp=False)
            if result.converged:
                break
        except:
            continue
    
    # Get slope k (fixed effect of dt)
    if result is not None and result.converged:
        k = float(result.params.get("dt", np.nan))
    else:
        # Fallback OLS for slope if MixedLM fails
        ols = smf.ols("S ~ dt", data=dfm).fit()
        k = float(ols.params.get("dt", np.nan))
    
    if not np.isfinite(k) or abs(k) < 1e-8:
        # if slope is (near) zero, default to small positive slope to avoid blow-up
        if verbose:
            print("[MixedLM] slope near zero; using fallback k=1.0")
        k = 1.0
    
    # Random intercepts u_i
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
