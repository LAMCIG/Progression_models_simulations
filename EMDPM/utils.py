import numpy as np
from scipy.integrate import solve_ivp

def solve_system2(x0: np.ndarray, f: np.ndarray, K: np.ndarray, t_span: np.ndarray, scalar_K: float = 1.0) -> np.ndarray: #, alpha: float = 1.0) -> np.ndarray:
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

def solve_system(x0: np.ndarray, f: np.ndarray, K: np.ndarray, t_span: np.ndarray,
                 scalar_K: float = 1.0, kappa: float = 0.0) -> np.ndarray:
    """
    Solve dx/dt = (I - diag(x)) ( (scalar_K * K + kappa * I) x + f ),   x(0) = 0
    Returns x with shape (n_biomarkers, len(t_span))
    """
    n = x0.size
    I = np.eye(n)

    def rhs(t, x):
        Ax = (scalar_K * (K @ x)) + (kappa * x) + f
        return (I - np.diag(x)) @ Ax

    sol = solve_ivp(rhs, (t_span[0], t_span[-1]), x0, t_eval=t_span, method="LSODA")
    return sol.y  # shape (n, T)

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
        
def compute_sK_sens(x0: np.ndarray, f: np.ndarray, K: np.ndarray, t_span: np.ndarray,
                    scalar_K: float, kappa: float = 0.0, t_eval=None):
    """
    Forward sensitivity S_alpha = ∂x/∂scalar_K for
        dx/dt = (I - diag(x))((scalar_K*K + kappa*I)x + f)

    dS/dt = -diag(S)((scalar_K*K + kappa*I)x + f)
            + (I - diag(x)) (K x + (scalar_K*K + kappa*I) S)
    """
    n = x0.size
    I = np.eye(n)
    if t_eval is None:
        t_eval = t_span

    def aug_rhs(t, y):
        x = y[:n]
        S = y[n:].reshape(n)
        Ax = (scalar_K * (K @ x)) + (kappa * x) + f
        # main dynamics
        dxdt = (I - np.diag(x)) @ Ax
        # sensitivity wrt scalar_K (alpha)
        Kx = K @ x
        dSdt = -np.diag(S) @ Ax + (I - np.diag(x)) @ (Kx + (scalar_K * K + kappa * I) @ S)
        return np.concatenate([dxdt, dSdt])

    y0 = np.concatenate([x0, np.zeros_like(x0)])
    sol = solve_ivp(aug_rhs, (t_span[0], t_span[-1]), y0, t_eval=t_eval, method="LSODA")
    X = sol.y[:n, :]
    S_alpha = sol.y[n:, :]

    return sol.t, X, S_alpha


def compute_kappa_sens(x0: np.ndarray, f: np.ndarray, K: np.ndarray, t_span: np.ndarray,
                       scalar_K: float, kappa: float = 0.0, t_eval=None):
    """
    Forward sensitivity S_kappa = ∂x/∂kappa for
        dx/dt = (I - diag(x))((scalar_K*K + kappa*I)x + f)

    dS/dt = -diag(S)((scalar_K*K + kappa*I)x + f)
            + (I - np.diag(x)) (x + (scalar_K*K + kappa*I) S)
    """
    n = x0.size
    I = np.eye(n)
    if t_eval is None:
        t_eval = t_span

    def aug_rhs(t, y):
        x = y[:n]
        S = y[n:].reshape(n)
        Ax = (scalar_K * (K @ x)) + (kappa * x) + f
        dxdt = (I - np.diag(x)) @ Ax
        dSdt = -np.diag(S) @ Ax + (I - np.diag(x)) @ (x + (scalar_K * K + kappa * I) @ S)
        return np.concatenate([dxdt, dSdt])

    y0 = np.concatenate([x0, np.zeros_like(x0)])
    sol = solve_ivp(aug_rhs, (t_span[0], t_span[-1]), y0, t_eval=t_eval, method="LSODA")
    X = sol.y[:n, :]
    S_kappa = sol.y[n:, :]

    return sol.t, X, S_kappa
