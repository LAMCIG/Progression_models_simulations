import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from .utils import solve_system

# Log-normal prior center for scalar_K (penalty pulls toward this value)
SCALAR_K_CENTER = 0.1
# Lower bound for scalar_K in log-normal penalty/gradient to avoid log(0) and divide-by-zero
SCALAR_K_MIN = 1e-12

def theta_s_loss(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                 K: np.ndarray, t_span: np.ndarray, f: np.ndarray,
                 lambda_s: float = 0.0, lambda_scalar: float = 0.0) -> float:
    """
    Computes loss for optimizing global scaling parameter s and scalar_K.
    
    Parameters
    ----------
    params : np.ndarray
        Current guess: [s, scalar_K] where s is (n_biomarkers,) and scalar_K is scalar.
    t_obs : np.ndarray
        Observation times.
    x_obs : np.ndarray
        Observed biomarker values (shape: n_obs x n_biomarkers).
    K : np.ndarray
        Connectivity matrix.
    t_span : np.ndarray
        Time points for trajectory simulation.
    f : np.ndarray
        Fixed forcing term (shape: n_biomarkers).
    lambda_s : float
        Regularization strength for s (optional).
    lambda_scalar : float
        Regularization strength for scalar_K (optional).
        
    Returns
    -------
    float
        Loss value.
    """
    n_biomarkers = x_obs.shape[1]
    s = params[:n_biomarkers]
    scalar_K = params[-1]
    
    x0 = np.zeros(n_biomarkers)
    x = solve_system(x0, f, K, t_span, scalar_K)
    x_scaled = s[:, None] * x
    
    x_pred = np.zeros_like(x_obs)  # (n_obs, n_biomarkers)
    for j in range(n_biomarkers):
        x_pred[:, j] = np.interp(t_obs, t_span, x_scaled[j])
    
    residuals = x_obs - x_pred
    # loss = np.sum(residuals**2) + lambda_s * np.sum(s**2) + lambda_scalar * scalar_K**2  # old L2
    scalar_K_safe = max(scalar_K, SCALAR_K_MIN)
    lognorm_penalty = 0.5 * lambda_scalar * (np.log(scalar_K_safe) - np.log(SCALAR_K_CENTER)) ** 2
    loss = np.sum(residuals**2) + lambda_s * np.sum(s**2) + lognorm_penalty

    return loss

def theta_s_loss_jac(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                     K: np.ndarray, t_span: np.ndarray, f: np.ndarray,
                     lambda_s: float = 0.0, lambda_scalar: float = 0.0) -> tuple:
    """
    Computes loss and gradient for optimizing global scaling parameter s and scalar_K.
    
    Parameters
    ----------
    params : np.ndarray
        Current guess: [s, scalar_K] where s is (n_biomarkers,) and scalar_K is scalar.
    t_obs : np.ndarray
        Observation times.
    x_obs : np.ndarray
        Observed biomarker values (shape: n_obs x n_biomarkers).
    K : np.ndarray
        Connectivity matrix.
    t_span : np.ndarray
        Time points for trajectory simulation.
    f : np.ndarray
        Fixed forcing term (shape: n_biomarkers).
    lambda_s : float
        Regularization strength for s (optional).
    lambda_scalar : float
        Regularization strength for scalar_K (optional).
        
    Returns
    -------
    tuple
        Loss scalar and gradient vector [grad_s, grad_scalar_K].
    """
    n_biomarkers = x_obs.shape[1]
    s = params[:n_biomarkers]
    scalar_K = params[-1]
    
    x0 = np.zeros(n_biomarkers)
    x = solve_system(x0, f, K, t_span, scalar_K)
    x_scaled = s[:, None] * x
    
    x_pred = np.zeros_like(x_obs)  # (n_obs, n_biomarkers)
    for j in range(n_biomarkers):
        x_pred[:, j] = np.interp(t_obs, t_span, x_scaled[j])
    
    residuals = x_obs - x_pred
    # loss = np.sum(residuals**2) + lambda_s * np.sum(s**2) + lambda_scalar * scalar_K**2  # old L2
    scalar_K_safe = max(scalar_K, SCALAR_K_MIN)
    lognorm_penalty = 0.5 * lambda_scalar * (np.log(scalar_K_safe) - np.log(SCALAR_K_CENTER)) ** 2
    loss = np.sum(residuals**2) + lambda_s * np.sum(s**2) + lognorm_penalty

    # Gradient with respect to s
    # d/ds (s * x) = x, so gradient of residual^2 is -2 * residuals * x_pred / s
    # But x_pred = s * x_interp, so we need to use x_interp (unscaled)
    x_interp = np.zeros_like(x_obs)
    for j in range(n_biomarkers):
        x_interp[:, j] = np.interp(t_obs, t_span, x[j])
    
    grad_s = -2 * np.sum(residuals * x_interp, axis=0) + 2 * lambda_s * s
    
    # Gradient with respect to scalar_K
    # Need to compute d/d(scalar_K) of x
    from scipy.interpolate import CubicSpline
    from scipy.integrate import cumulative_simpson
    
    Kx_plus_f = (K @ x) + f[:, None]  # n_biomarkers x len(t_span)
    scalar_expr = (1 - x) * Kx_plus_f
    
    scalar_obs = np.zeros_like(x_obs)
    for i in range(n_biomarkers):
        interp_fn = CubicSpline(t_span, scalar_expr[i], extrapolate=True)
        scalar_obs[:, i] = interp_fn(t_obs)
    
    # Need to account for s scaling: d/ds(s*x) with respect to scalar_K
    # grad_scalar_K = -2 * np.sum(residuals * (scalar_obs * s[None, :])) + 2 * lambda_scalar * scalar_K  # old L2
    lognorm_grad = lambda_scalar * (np.log(scalar_K_safe) - np.log(SCALAR_K_CENTER)) / scalar_K_safe
    grad_scalar_K = -2 * np.sum(residuals * (scalar_obs * s[None, :])) + lognorm_grad

    grad = np.concatenate([grad_s, [grad_scalar_K]])
    
    return loss, grad


def theta_s_loss_multi(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                       K: np.ndarray, t_span: np.ndarray, cluster_f: list,
                       observation_assignments: np.ndarray,
                       lambda_s: float = 0.0, lambda_scalar: float = 0.0) -> float:
    """
    Loss for s and scalar_K when each observation uses its assigned subtype's f.
    observation_assignments[i] = subtype index for the i-th observation row.
    """
    n_biomarkers = x_obs.shape[1]
    n_obs = x_obs.shape[0]
    s = params[:n_biomarkers]
    scalar_K = params[-1]
    x0 = np.zeros(n_biomarkers)

    # One trajectory per subtype (unscaled then scale by s)
    n_subtypes = len(cluster_f)
    x_scaled_list = []
    for k in range(n_subtypes):
        f_k = np.ravel(cluster_f[k])
        x_k = solve_system(x0, f_k, K, t_span, scalar_K)
        x_scaled_k = s[:, None] * x_k
        x_scaled_list.append(x_scaled_k)

    # Predicted value per observation: vectorize by subtype (one interp per subtype per biomarker)
    x_pred = np.zeros_like(x_obs)
    for k in range(n_subtypes):
        mask_k = (observation_assignments == k)
        if not np.any(mask_k):
            continue
        t_k = t_obs[mask_k]
        for j in range(n_biomarkers):
            x_pred[mask_k, j] = np.interp(t_k, t_span, x_scaled_list[k][j])

    residuals = x_obs - x_pred
    # loss = np.sum(residuals ** 2) + lambda_s * np.sum(s ** 2) + lambda_scalar * scalar_K ** 2  # old L2
    scalar_K_safe = max(scalar_K, SCALAR_K_MIN)
    lognorm_penalty = 0.5 * lambda_scalar * (np.log(scalar_K_safe) - np.log(SCALAR_K_CENTER)) ** 2
    loss = np.sum(residuals ** 2) + lambda_s * np.sum(s ** 2) + lognorm_penalty
    return loss


def theta_s_loss_jac_multi(params: np.ndarray, t_obs: np.ndarray, x_obs: np.ndarray,
                           K: np.ndarray, t_span: np.ndarray, cluster_f: list,
                           observation_assignments: np.ndarray,
                           lambda_s: float = 0.0, lambda_scalar: float = 0.0) -> tuple:
    """
    Loss and gradient for s and scalar_K when each observation uses its assigned subtype's f.
    """
    n_biomarkers = x_obs.shape[1]
    n_obs = x_obs.shape[0]
    s = params[:n_biomarkers]
    scalar_K = params[-1]
    x0 = np.zeros(n_biomarkers)
    n_subtypes = len(cluster_f)

    # Trajectories per subtype (unscaled and scaled)
    x_list = []
    x_scaled_list = []
    for k in range(n_subtypes):
        f_k = np.ravel(cluster_f[k])
        x_k = solve_system(x0, f_k, K, t_span, scalar_K)
        x_list.append(x_k)
        x_scaled_k = s[:, None] * x_k
        x_scaled_list.append(x_scaled_k)

    # Precompute CubicSplines for scalar_K gradient: one per (subtype, biomarker)
    splines_scalar = []
    for k in range(n_subtypes):
        x_k = x_list[k]
        f_k = np.ravel(cluster_f[k])
        Kx_plus_f = (K @ x_k) + f_k[:, None]
        scalar_expr = (1 - x_k) * Kx_plus_f
        splines_scalar.append([
            CubicSpline(t_span, scalar_expr[j], extrapolate=True)
            for j in range(n_biomarkers)
        ])

    # Predictions and residuals: vectorize by subtype
    x_pred = np.zeros_like(x_obs)
    for k in range(n_subtypes):
        mask_k = (observation_assignments == k)
        if not np.any(mask_k):
            continue
        t_k = t_obs[mask_k]
        for j in range(n_biomarkers):
            x_pred[mask_k, j] = np.interp(t_k, t_span, x_scaled_list[k][j])

    residuals = x_obs - x_pred
    # loss = np.sum(residuals ** 2) + lambda_s * np.sum(s ** 2) + lambda_scalar * scalar_K ** 2  # old L2
    scalar_K_safe = max(scalar_K, SCALAR_K_MIN)
    lognorm_penalty = 0.5 * lambda_scalar * (np.log(scalar_K_safe) - np.log(SCALAR_K_CENTER)) ** 2
    loss = np.sum(residuals ** 2) + lambda_s * np.sum(s ** 2) + lognorm_penalty

    # Gradient for s: vectorize by subtype (one interp per subtype per biomarker, then sum)
    grad_s = np.zeros(n_biomarkers)
    for k in range(n_subtypes):
        mask_k = (observation_assignments == k)
        if not np.any(mask_k):
            continue
        t_k = t_obs[mask_k]
        x_interp_k = np.zeros((np.sum(mask_k), n_biomarkers))
        for j in range(n_biomarkers):
            x_interp_k[:, j] = np.interp(t_k, t_span, x_list[k][j])
        grad_s = grad_s - 2 * np.sum(residuals[mask_k] * x_interp_k, axis=0)
    grad_s = grad_s + 2 * lambda_s * s

    # Gradient for scalar_K: evaluate prebuilt splines per observation
    grad_scalar_K = 0.0
    for i in range(n_obs):
        a_i = int(observation_assignments[i])
        scalar_obs_i = np.array([splines_scalar[a_i][j](t_obs[i]) for j in range(n_biomarkers)])
        grad_scalar_K = grad_scalar_K - 2 * np.sum(residuals[i] * s * scalar_obs_i)
    # grad_scalar_K = grad_scalar_K + 2 * lambda_scalar * scalar_K  # old L2
    lognorm_grad = lambda_scalar * (np.log(scalar_K_safe) - np.log(SCALAR_K_CENTER)) / scalar_K_safe
    grad_scalar_K = grad_scalar_K + lognorm_grad

    grad = np.concatenate([grad_s, [grad_scalar_K]])
    return loss, grad


def fit_theta_globals(X_obs: np.ndarray, dt_obs: np.ndarray, ids: np.ndarray, K: np.ndarray,
                      t_span: np.ndarray, use_jacobian: bool,
                      f: np.ndarray = None,
                      beta_pred: np.ndarray = None,
                      s_guess: np.ndarray = None, scalar_K_guess: float = None,
                      lambda_s: float = 0.0, lambda_scalar: float = 0.0,
                      rng: np.random.Generator = None,
                      assignments: np.ndarray = None,
                      cluster_f: list = None) -> tuple:
    """
    Optimizes global scaling parameter s (supremum) and scalar_K for all patients.
    If assignments and cluster_f are provided, uses assignment-aware loss (each observation
    predicted with its assigned subtype's f). Otherwise uses single f for all (legacy).
    """
    if rng is None:
        rng = np.random.default_rng(75)

    unique_ids = np.unique(ids)
    id_to_index = {pid: i for i, pid in enumerate(unique_ids)}
    index_array = np.array([id_to_index[i] for i in ids])  # shape: (n_obs,)
    t_pred = dt_obs + beta_pred[index_array]

    n_biomarkers = X_obs.shape[1]

    # Initial guesses if None
    if s_guess is None:
        s_guess = np.ones(n_biomarkers)
    if scalar_K_guess is None:
        scalar_K_guess = 1.0

    initial_params = np.concatenate([s_guess, [scalar_K_guess]])
    bounds = [(0.0, np.inf)] * n_biomarkers + [(0.0, np.inf)]

    use_multi = (assignments is not None and cluster_f is not None)
    if use_multi:
        observation_assignments = assignments[index_array]  # assignment per observation row
        if use_jacobian:
            loss_function = theta_s_loss_jac_multi
        else:
            loss_function = theta_s_loss_multi
        args = (t_pred, X_obs, K, t_span, cluster_f, observation_assignments, lambda_s, lambda_scalar)
    else:
        if f is None:
            raise ValueError("Must provide either f or (assignments and cluster_f).")
        if use_jacobian:
            loss_function = theta_s_loss_jac
        else:
            loss_function = theta_s_loss
        args = (t_pred, X_obs, K, t_span, f, lambda_s, lambda_scalar)

    result = minimize(
        loss_function,
        initial_params,
        args=args,
        method="L-BFGS-B",
        jac=use_jacobian,
        bounds=bounds
    )

    fitted_params = result.x
    s_fit = fitted_params[:n_biomarkers]
    scalar_K_fit = fitted_params[-1]

    return s_fit, scalar_K_fit
