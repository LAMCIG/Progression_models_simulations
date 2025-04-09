# TODO: get cognitive score implemented into beta optimization
# TODO: implement K_ij brain region into theta optimization
# TODO: implement jacobian for each theta 
# TODO: Use random state instead

import numpy as np
from scipy.integrate import solve_ivp

# TODO: Migrate the seed to the class or the 

def get_adjacency_matrix(connectivity_matrix_type, n_biomarkers, seed):
    """
    Generate a connectivity matrix representing brain region interactions.

    Parameters
    ----------
    connectivity_matrix_type : str
        Type of connectivity ('offdiag' or 'random_offdiag').
    n_biomarkers : int
        Number of biomarkers (dimensions of the matrix).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    matrix : np.ndarray
        Connectivity matrix of shape (n_biomarkers, n_biomarkers).
    """
    if connectivity_matrix_type == 'offdiag':
        matrix = np.zeros((n_biomarkers, n_biomarkers))
        for i in range(n_biomarkers - 1):
            matrix[i, i + 1] = 1
            matrix[i + 1, i] = 1
        return matrix

    if connectivity_matrix_type == 'random_offdiag':
        np.random.seed(seed)
        matrix = np.zeros((n_biomarkers, n_biomarkers))

        # fully connect first offdiag
        first_off_diag_values = np.random.random(n_biomarkers - 1)
        np.fill_diagonal(matrix[1:], first_off_diag_values)
        np.fill_diagonal(matrix[:, 1:], first_off_diag_values)

        # further off diags
        for offset in range(2, 3):  # just 2nd and 3rd for now
            sparsity = 0.1 * offset
            scale = 1 / offset
            random_values = np.random.random(size=n_biomarkers - offset) * scale

            # apply sparsity mask (higher offset = more zeros)
            mask = np.random.rand(n_biomarkers - offset) > sparsity
            random_values *= mask  # Zero out some connections

            np.fill_diagonal(matrix[offset:], random_values)
            np.fill_diagonal(matrix[:, offset:], random_values)

        matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

        return matrix

    else:
        raise ValueError("Unknown connectivity matrix type")

def multi_logistic_deriv_force(t, x, K, f):
    """
    Compute the time derivative dx/dt for the multivariate logistic system.

    Parameters
    ----------
    t : float
        Time point (unused, but required by ODE solver).
    x : np.ndarray
        State vector of biomarkers.
    K : np.ndarray
        Connectivity matrix.
    f : np.ndarray
        External forcing vector.
    alpha: float
        Scalar multiplier on K

    Returns
    -------
    dx_dt : np.ndarray
        Time derivative of the system.
    """
    x = np.maximum(x, 0)
    return (np.eye(K.shape[0]) - np.diag(x)) @ (K @ x) + f

def generate_logistic_model(n_biomarkers=10, step=0.1, t_max=10, connectivity_matrix_type='random_offdiag', seed = 75):
    """
    Generate a synthetic multivariate logistic progression model.
    dx/dt = (I - diag(x)) @ (K @ x + f)

    Parameters
    ----------
    n_biomarkers : int, optional
        Number of biomarkers to simulate.
    step : float, optional
        Time step for simulation.
    t_max : float, optional
        Maximum time value for simulation.
    connectivity_matrix_type : str, optional
        Type of connectivity matrix to use.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    t : np.ndarray
        Time points of the model simulation.
    y : np.ndarray
        Simulated biomarker trajectories (n_biomarkers x len(t)).
    K : np.ndarray
        Connectivity matrix used in the simulation.
    """
    t_eval = np.arange(0, t_max, step)

    x0 = np.zeros(n_biomarkers)
    f = np.random.gamma(shape=2, scale=0.005, size=n_biomarkers)

    x0[x0 < 0.01] = 0  # probably unnecessary but just to get rid of some small values
    f[f < 0.01] = 0

    # TODO: add to verbose feedback
    print(f"true x0: {x0}")
    print(f"true f: {f}")

    K = get_adjacency_matrix(connectivity_matrix_type, n_biomarkers, seed)
    sol = solve_ivp(multi_logistic_deriv_force, t_span=[0, t_max], y0=x0, args=(K, f),
                    t_eval=t_eval, method="DOP853")

    return sol.t, sol.y, K