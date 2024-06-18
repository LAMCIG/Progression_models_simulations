import numpy as np

class ODEGenerator:
    def __init__(self, n_biomarkers, step=0.05, n_steps=200, neg_frac=0.01):
        self.n_biomarkers = n_biomarkers
        self.step = step
        self.n_steps = n_steps
        self.neg_frac = neg_frac

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

        return K

    def multi_logistic_deriv_force(self, K, f, x):
        return np.dot(np.eye(K.shape[0]) - np.diag(x), np.dot(K, x) + f)

    def multi_logistic_sym_force(self):
        n = self.n_biomarkers
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
