import numpy as np
from scipy.integrate import odeint
class ODEGenerator:
    def __init__(self, n_biomarker_stages, model_type: str = 'logistic', step=0.05, n_steps=200, random_state = 10):
        self.n_biomarker_stages = n_biomarker_stages
        self.step = step
        self.n_steps = n_steps
        self.model_type = model_type
        
        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)
        
        # Attributes
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
        np.random.RandomState(self.random_state) # fixing random state
        A = self.rng.rand(n, n)
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
        neg_frac = 0.01

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
        dt = t[1] - t[0]  # time step
        x0 = np.zeros(n)
        x0[0] = 0.05
        
        #A = self.random_connectivity_matrix(n, 1.2, 0.1, False)
        A = np.matrix([[0,1,1],
                        [1,0,1],
                        [1,1,0]])
        
        H = np.diag(np.sum(A, axis=0)) - A
        
        k_ij = np.random.normal(loc=0.05, scale=1, size=3)
        gamma = np.ones(3) * 0.6
        eta = np.ones(3) * 0.9
        
        # k_ij = self.rng.uniform(0.05, 1, n) 
        # gamma = self.rng.uniform(0.4, 0.8, n)
        # eta = self.rng.uniform(1.0, 1.5, n)
        
        print(f'k_ij: {k_ij}\ngamma: {gamma}\neta:{eta}')
        
        l1 = 3
        l2 = 3

        # clipping range to avoid overflow
        clip_min = -500
        clip_max = 500

        def acp_equations(f, tau, A, H, k_ij, gamma, eta, l1, l2):
            exp_1 = np.clip(-l1 * (f - gamma), clip_min, clip_max)
            exp_2 = np.clip(l2 * (f - eta), clip_min, clip_max)
            
            K_ACP = k_ij / ((1 + np.exp(exp_1)) * (1 + np.exp(exp_2)))
            R_ACP = k_ij / (1 + np.exp(exp_2))
            dfdtau = np.dot((A * K_ACP), np.dot(H, f)) + R_ACP * f
            return dfdtau

        # solution array
        x = np.zeros((n, len(t)))
        x[:, 0] = x0

        # forward euler method
        for i in range(1, len(t)):
            dx_dt = acp_equations(x[:, i-1], t[i-1], A, H, k_ij, gamma, eta, l1, l2)
            x[:, i] = x[:, i-1] + dx_dt * dt
            x[:, i] = np.maximum(x[:, i], 0)  # enforce non-negativity

        return t, x

#%% DIFFUSION MODEL
    def diffusion_model(self):
        np.random.RandomState(self.random_state)
        n = self.n_biomarker_stages
        t = np.linspace(0, 100, 200)
        step = t[1] - t[0]  # time step
        x = np.zeros((n, len(t)))
        
        x[:, 0] = np.ones(n) * 0.5 + np.random.normal(0, 0.01, n)
        x[0, 0] = 0.1  # set the initial condition for the first biomarker
        
        A = self.random_connectivity_matrix(n, 1.2, 0.1, False)
        H = np.diag(np.sum(A, axis=0)) - A

        k_ij = np.random.uniform(0.5, 2, n)
        K = np.diag(k_ij)
        
        # euler
        for i in range(1, len(t)):
            dfdtau = -np.dot(np.dot(A * K, H), x[:, i-1])
            x[:, i] = x[:, i-1] + dfdtau * step
        return t, x

#%% REACTION DIFFUSION (RD)
    def reaction_diffusion_model(self):
        n = self.n_biomarker_stages
        step = self.step
        n_steps = self.n_steps
        t = np.linspace(0, 100, n_steps)
        x0 = np.ones(n) * 0.05
        x0[0] = 0.05

        A = self.random_connectivity_matrix(n, 1.2, 0.1, False)
        H = np.diag(np.sum(A, axis=0)) - A

        k_ij = np.random.uniform(0.2, 1, n)
        K = np.diag(k_ij)
        nu = np.random.uniform(1, 2, n)  # maximal concentration threshold
        R = np.random.uniform(0.1, 0.3, n)  # aggregation rate

        x = np.zeros((n, n_steps))
        x[:, 0] = x0

        for i in range(1, n_steps):
            diffusion_term = np.dot((A * K), np.dot(H, x[:, i-1]))
            reaction_term = R * x[:, i-1] * (nu - x[:, i-1])
            dx_dt = diffusion_term + reaction_term
            x[:, i] = x[:, i-1] + step * np.maximum(dx_dt, 0)  # forward euler step

        return t, x

#%%
    def get_connectivity_matrix(self):
        return self.connectivity_matrix
    
