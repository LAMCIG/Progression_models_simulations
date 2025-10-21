# TODO: get cognitive score implemented into beta optimization
# TODO: implement K_ij brain region into theta optimization
# TODO: implement jacobian for each theta 
# TODO: Use random state instead

import numpy as np
from scipy.integrate import solve_ivp

# TODO: Migrate the seed to the class or the 

def get_adjacency_matrix(connectivity_matrix_type, n_biomarkers, rng):
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
        matrix = np.zeros((n_biomarkers, n_biomarkers))

        # fully connect first offdiag
        first_off_diag_values = rng.random(n_biomarkers - 1)
        np.fill_diagonal(matrix[1:], first_off_diag_values)
        np.fill_diagonal(matrix[:, 1:], first_off_diag_values)

        # further off diags
        for offset in range(2, 3):  # just 2nd and 3rd for now
            sparsity = 0.1 * offset
            scale = 1 / offset
            random_values = rng.random(size=n_biomarkers - offset) * scale

            # apply sparsity mask (higher offset = more zeros)
            mask = rng.random(n_biomarkers - offset) > sparsity
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

    dx/dt = (I - diag(x)) @ (K @ x + f)

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
    return (np.eye(K.shape[0]) - np.diag(x)) @ (K @ x + f)

def generate_logistic_model(n_biomarkers=10,
                            step=0.1,
                            t_max=10,
                            connectivity_matrix_type='random_offdiag',
                            seed = 75,
                            scalar_K = 1.0,
                            rng = None,
                            K = None,
                            f = None):
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
    if rng is None:
        rng = np.random.default_rng(seed)
    
    t_eval = np.arange(0, t_max, step)
    x0 = np.zeros(n_biomarkers)
    
    if f is None:
        f = rng.gamma(shape=2, scale=0.005, size=n_biomarkers)
        f[f < 0.005] = 0.0 # remove small values
    else:
        f = np.asarray(f, float)
        assert f.shape == (n_biomarkers,), "forcing term f must have shape (n_biomarkers,)"
    
    if K is None:
        K = get_adjacency_matrix(connectivity_matrix_type, n_biomarkers, rng)
    else:
        K = np.asarray(K, float)
        assert K.shape == (n_biomarkers, n_biomarkers), "connectome K must be shape (n_biomarkers, n_biomarkers)"   
       
    if scalar_K is None:
        scalar_K = 1.0

    #x0[x0 < 0.005] = 0  # probably unnecessary but just to get rid of some small values
    f[f < 0.005] = 0
    
    K = get_adjacency_matrix(connectivity_matrix_type, n_biomarkers, rng)
    K_scaled = scalar_K * K
    
    sol = solve_ivp(multi_logistic_deriv_force,
                    t_span=[0, t_max],
                    y0=x0,
                    args=(K_scaled, f),
                    t_eval=t_eval,
                    method="LSODA")

    return sol.t, sol.y, K, x0, f, scalar_K

def create_patient_list(X_obs, ids, dt, cog, initial_beta=None):
    unique_ids = np.unique(ids)
    id_to_index = {pid: idx for idx, pid in enumerate(unique_ids)}

    patient_list = []
    for pid in unique_ids:
        mask = (ids == pid)
        patient_data = {
            "id": pid,
            "X_obs": X_obs[mask],
            "dt": dt[mask],
            "cog": cog[mask],
        }
        if initial_beta is not None:
            patient_data["initial_beta"] = initial_beta[id_to_index[pid]]
        patient_list.append(patient_data)

    return patient_list