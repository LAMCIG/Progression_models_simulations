import numpy as np
from scipy.integrate import solve_ivp

def solve_system(x0: np.ndarray, f: np.ndarray, K: np.ndarray, t_span: np.ndarray) -> np.ndarray:
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
    def ode_system(t, x):
        return (np.eye(K.shape[0]) - np.diag(x)) @ (K @ x) + f

    def jacobian_ode(t, x):
        Kx = K @ x
        J = (1 - x)[:, None] * K
        return J

    sol = solve_ivp(
        ode_system,
        [t_span[0], t_span[-1]],
        x0,
        t_eval=t_span,
        method="LSODA",
        jac=jacobian_ode
    )

    return sol.y
