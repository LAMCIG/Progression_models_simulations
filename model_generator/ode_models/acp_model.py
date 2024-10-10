import numpy as np
from model_generator.base_disease_model import BaseDiseaseModel  # Assuming this is your base model
from model_generator.biomarker_utils import get_adjacency_matrix, compute_laplacian_matrix  # Assuming you have these utility functions

class ACPModel(BaseDiseaseModel):
    def __init__(self, matrix_type='Tridiagonal', **params):
        super().__init__(**params)
        self.matrix_type = matrix_type  # Set the type of adjacency matrix

    def acp_equations(self, f, tau, A, H, k_ij, gamma, eta, l1, l2):
        exp_1 = -l1 * (f - gamma)
        exp_2 = l2 * (f - eta)

        K_ACP = k_ij / ((1 + np.exp(exp_1)) * (1 + np.exp(exp_2)))
        R_ACP = k_ij / (1 + np.exp(exp_2))
        dfdtau = np.dot((A * K_ACP), np.dot(H, f)) + np.multiply(R_ACP, f)
        return dfdtau

    def fit(self, X=None, y=None):
        n_stages = self.params.get('n_stages', 10)
        l1_mean = self.params.get('l1_mean', 3.0)
        l2_mean = self.params.get('l2_mean', 3.0)
        gamma_mean = self.params.get('gamma_mean', 0.6)
        eta_mean = self.params.get('eta_mean', 0.9)
        k_ij_value = self.params.get('k_ij_value', 0.5)
        random_state = self.params.get('random_state', 10)

        np.random.seed(random_state)
        A = get_adjacency_matrix(self.matrix_type, n_stages)  # adjacency matrix
        H = compute_laplacian_matrix(A)  # laplacian matrix

        t = np.linspace(self.start_time, self.end_time, self.steps)
        dt = t[1] - t[0]  # Time step
        x0 = np.zeros(n_stages)
        x0[0] = 0.05  # Initial condition for the first biomarker

        # params for eqn
        l1 = np.random.normal(loc=l1_mean, scale=1, size=n_stages)
        l2 = np.random.normal(loc=l2_mean, scale=1, size=n_stages)
        gamma = np.random.normal(loc=gamma_mean, scale=0.1, size=n_stages)
        eta = np.random.normal(loc=eta_mean, scale=0.1, size=n_stages)
        k_ij = np.random.normal(loc=k_ij_value, scale=0.1, size=n_stages)

        self.model_values = np.zeros((n_stages, len(t)))
        self.model_values[:, 0] = x0  # Set initial condition

        # forward euler
        for i in range(1, len(t)):
            dx_dt = self.acp_equations(self.model_values[:, i-1], t[i-1], A, H, k_ij, gamma, eta, l1, l2)
            self.model_values[:, i] = self.model_values[:, i-1] + dx_dt * dt
            self.model_values[:, i] = np.maximum(self.model_values[:, i], 0)  # Enforce non-negativity

        # dynamically adjust the time if necessary (using the base model method)
        adjusted_time_points = self._dynamic_time_adjustment(self.model_values)
        if len(adjusted_time_points) > len(t):
            # recompute model values for the adjusted time points
            self.model_values = np.zeros((n_stages, len(adjusted_time_points)))
            self.model_values[:, 0] = x0  # Reapply initial condition
            for i in range(1, len(adjusted_time_points)):
                dx_dt = self.acp_equations(self.model_values[:, i-1], adjusted_time_points[i-1], A, H, k_ij, gamma, eta, l1, l2)
                self.model_values[:, i] = self.model_values[:, i-1] + dx_dt * dt
                self.model_values[:, i] = np.maximum(self.model_values[:, i], 0)  # enforce non-negativity

        return self
