import numpy as np
from scipy.integrate import odeint
class ODEGenerator:
    def __init__(self, n_biomarker_stages, model_type: str = 'logistic', step=0.05, n_steps=200, neg_frac=0.01):
        self.n_biomarker_stages = n_biomarker_stages
        self.step = step
        self.n_steps = n_steps
        self.neg_frac = neg_frac
        self.model_type = model_type
        self.connectivity_matrix = None
        self.time_points = np.linspace(0, 100, 1000)
        
    def generate_model(self):
        if self.model_type == 'logistic':
            return self.multi_logistic_sym_force()
        elif self.model_type == 'acp':
            return self.acp_model()
        elif self.model_type == 'diffusion':
            return self.diffusion_model()
        elif self.model_type == 'reaction_diffusion':
            return self.reaction_diffusion_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def random_connectivity_matrix(self, n, med_frac, source_rate, all_source_connections):
        A = np.random.rand(n, n)
        A = np.dot(A.T, A)
        np.fill_diagonal(A, 0)
        K = A.copy()

        for i in range(n):
            loc_thresh = min(med_frac * np.median(A[i, 1:]), max(A[i, 1:] * 0.99))
            ind = A[i, 1:] < loc_thresh
            ind = np.insert(ind, 0, False)
            K[i, ind] = 0
            K[ind, i] = 0

        S = np.sum(K > 0.0, axis=0)
        for i in range(n):
            if S[i] == 0 or (i == 1 and S[i] < 2):
                if i != 1:
                    val, ind = max((val, idx) for idx, val in enumerate(A[i, 1:]))
                    K[i, ind + 1] = val
                    K[ind + 1, i] = val
                else:
                    val, ind = max((val, idx) for idx, val in enumerate(A[i, 2:]))
                    K[i, ind + 2] = val
                    K[ind + 2, i] = val

        K /= np.max(K)
        if all_source_connections:
            K[0, :] = source_rate
            K[:, 0] = source_rate
        else:
            K[0, :] = 0.0
            K[:, 0] = 0.0

        K[0, 1] = source_rate
        K[1, 0] = source_rate

        self.connectivity_matrix = K
        return K
#%% LOGISTIC MODEL
    def multi_logistic_deriv_force(self, K, f, x):
        return np.dot(np.eye(K.shape[0]) - np.diag(x), np.dot(K, x) + f)

    def multi_logistic_sym_force(self):
        n = self.n_biomarker_stages
        step = self.step
        n_steps = self.n_steps
        neg_frac = self.neg_frac

        t0 = -neg_frac * n_steps * step
        t = np.arange(t0, (n_steps + 1) * step, step)
        x0 = np.zeros(n)
        f = np.zeros(n)
        f[0] = 0.01

        x = np.zeros((n, len(t)))
        zero_ind = np.where(t == 0)[0][0]
        x[:, zero_ind] = x0

        K = self.random_connectivity_matrix(n, 1.2, 0.1, 0)

        for i in range(zero_ind - 1, -1, -1):
            force = np.zeros_like(f)
            dx_dt = self.multi_logistic_deriv_force(K, force, x[:, i + 1])
            x[:, i] = np.maximum(0, x[:, i + 1] - dx_dt * step)

        for i in range(zero_ind, n_steps + zero_ind):
            force = np.exp(t[i] * f) - 1
            dx_dt = self.multi_logistic_deriv_force(K, force, x[:, i])
            x[:, i + 1] = x[:, i] + dx_dt * step

        return t, x

#%% ACP MODEL
    def acp_model(self):
        n = self.n_biomarker_stages
        t = np.linspace(0, 100, 200)
        x0 = np.zeros(n)
        x0[0] = 0.05
        
        A = self.random_connectivity_matrix(n, 1.2, 0.1, False)
        H = np.diag(np.sum(A, axis=0)) - A

        k_ij = np.random.uniform(0.05, 1, n) 
        gamma = np.random.uniform(0.4, 0.8, n)
        eta = np.random.uniform(1.0, 1.5, n)
        
        print(f'k_ij: {k_ij}\ngamma: {gamma}\neta:{eta}')
        
        l1 = 3
        l2 = 3

        def acp_equations(f, tau, A, H, k_ij, gamma, eta, l1, l2):
            K_ACP = k_ij / ((1 + np.exp(-l1 * (f - gamma))) * (1 + np.exp(l2 * (f - eta))))
            R_ACP = k_ij / (1 + np.exp(l2 * (f - eta)))
            dfdtau = np.dot((A * K_ACP), np.dot(H, f)) + R_ACP * f
            return np.maximum(dfdtau, 0)
        x = odeint(acp_equations, x0, t, args=(A, H, k_ij, gamma, eta, l1, l2))
        return t, x.T

#%% DIFFUSION MODEL
    def diffusion_model(self):
        n = self.n_biomarker_stages
        t = np.linspace(0, 100, 200)
        x0 = np.ones(n) * 0.5
        x0[0] = 0.5
            
        A = self.random_connectivity_matrix(n, 1.2, 0.1, False)
        H = np.diag(np.sum(A, axis=0)) - A

        k_ij = np.random.uniform(0.05, 1, n)
        K = np.diag(k_ij)

        def diffusion_equations(f, tau, A, H, K):
            dfdtau = -np.dot(np.dot(A * K, H), f)
            return dfdtau
            
        x = odeint(diffusion_equations, x0, t, args=(A, H, K))
        return t, x.T

#%% REACTION DIFFUSION (RD)
    def reaction_diffusion_model(self):
        n = self.n_biomarker_stages
        t = np.linspace(0, 100, 200)
        x0 = np.ones(n) * 0.5
        x0[0] = 0.5

        A = self.random_connectivity_matrix(n, 1.2, 0.1, False)
        H = np.diag(np.sum(A, axis=0)) - A

        k_ij = np.random.uniform(0.2, 1, n)
        K = np.diag(k_ij)
        nu = np.random.uniform(1, 2, n) # maximal concentration threshold
        R = np.random.uniform(0.1, 0.3, n) # agregation rate

        def reaction_diffusion_equations(f, tau, A, H, K, R, nu):
            diffusion_term = np.dot((A * K), np.dot(H, f))
            reaction_term = R * f * (nu - f)
            dfdtau = diffusion_term + reaction_term
            return np.maximum(dfdtau, 0)

        x = odeint(reaction_diffusion_equations, x0, t, args=(A, H, K, R, nu))
        return t, x.T

#%%
    def get_connectivity_matrix(self):
        return self.connectivity_matrix
    
