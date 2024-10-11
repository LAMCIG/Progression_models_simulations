import numpy as np
from model_generator.base_disease_model import BaseDiseaseModel
from model_generator.biomarker_utils import get_adjacency_matrix

class LogisticModel(BaseDiseaseModel):
    def __init__(self, connectivity_matrix_type='default', **params):
        super().__init__(**params)
        self.connectivity_matrix_type = connectivity_matrix_type

    def multi_logistic_deriv_force(self, K, f, x):
        return np.dot(np.eye(K.shape[0]) - np.diag(x), np.dot(K, x) + f)

    def fit(self, X=None, y=None):
        n_biomarkers = self.params.get('n_stages', 10)
        step = self.params.get('step', 0.1)
        n_steps = self.params.get('n_steps', 100)
        neg_frac = 0.01

        # set initial conditions
        t0 = -neg_frac * n_steps * step 
        t = np.arange(t0, (n_steps + 1) * step, step)
        x0 = np.zeros(n_biomarkers)
        f = np.zeros(n_biomarkers)
        f[0] = 0.01

        # store things here
        x = np.zeros((n_biomarkers, len(t)))

        zero_ind = np.where(t == 0)[0][0]
        x[:, zero_ind] = x0

        # should be able to swap to any connectivity matrix.
        K = get_adjacency_matrix(self.connectivity_matrix_type, n_biomarkers)
        self.connectivity_matrix = K
        
        # backward integration
        for i in range(zero_ind - 1, -1, -1):
            force = np.zeros_like(f)
            dx_dt = self.multi_logistic_deriv_force(K, force, x[:, i + 1])
            x[:, i] = np.maximum(0, x[:, i + 1] - dx_dt * step)

        # forward integration
        for i in range(zero_ind, n_steps + zero_ind):
            force = np.exp(t[i] * f) - 1
            dx_dt = self.multi_logistic_deriv_force(K, force, x[:, i])
            x[:, i + 1] = x[:, i] + dx_dt * step
            
        self.model_values = x

        return self
    
    def get_connectivity_matrix(self):
        return self.connectivity_matrix

        # # dynamically adjust the time
        # adjusted_time_points = self._dynamic_time_adjustment(self.model_values)
        # if len(adjusted_time_points) > len(t):
        #     self.model_values = np.zeros((n_biomarkers, len(adjusted_time_points)))
        #     self.model_values[:, zero_ind] = x0  # reset initial conditions

        #     # adjust forward integration
        #     for i in range(zero_ind, len(adjusted_time_points) + zero_ind):
        #         force = np.exp(adjusted_time_points[i] * f) - 1
        #         dx_dt = self.multi_logistic_deriv_force(K, force, self.model_values[:, i])
        #         if i + 1 < len(adjusted_time_points):  # I don't like this adjustment but it works
        #             self.model_values[:, i + 1] = self.model_values[:, i] + dx_dt * step

