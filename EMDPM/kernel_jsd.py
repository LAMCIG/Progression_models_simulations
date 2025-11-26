import numpy as np


class KernelJSD:
    """
    Kernel-based Jensen-Shannon Divergence for comparing beta distributions.
    Used to prevent beta values from clumping into narrow bands by encouraging
    separation between subtypes (for 2 subtypes).
    """

    def __init__(self, alpha, beta, value_range=(0, 1), bandwidth=None, n_bins=None):
        self.alpha = np.asarray(alpha)
        self.beta = np.asarray(beta)
        self.range = value_range
        
        # Auto-determine number of bins based on range if not specified
        if n_bins is None:
            range_size = value_range[1] - value_range[0]
            self.n_bins = max(20, int(range_size * 2))  # At least 20 bins, or 2 per unit
        else:
            self.n_bins = n_bins
        self.bins = np.linspace(value_range[0], value_range[1], self.n_bins)
        
        if bandwidth is None:
            combined = np.concatenate([self.alpha, self.beta])
            std = np.std(combined)
            n = len(combined)
            self.h = 0.9 * std * n**(-1/5)
        else:
            self.h = bandwidth
        self.eps = 1e-10
    
    def _gaussian_kernel(self, x):
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def _gaussian_kernel_derivative(self, x):
        return -x * self._gaussian_kernel(x)
    
    def _estimate_densities(self):
        dist_alpha = (self.bins[:, None] - self.alpha[None, :]) / self.h
        dist_beta = (self.bins[:, None] - self.beta[None, :]) / self.h
        P = np.mean(self._gaussian_kernel(dist_alpha), axis=1) / self.h
        Q = np.mean(self._gaussian_kernel(dist_beta), axis=1) / self.h
        P = P / (P.sum() + self.eps)
        Q = Q / (Q.sum() + self.eps)
        return P, Q
    
    def jsd(self):
        """Compute Jensen-Shannon Divergence between alpha and beta distributions."""
        P, Q = self._estimate_densities()
        P = P + self.eps
        Q = Q + self.eps
        M = (P + Q) / 2
        kl_pm = np.sum(P * np.log(P / M))
        kl_qm = np.sum(Q * np.log(Q / M))
        return 0.5 * (kl_pm + kl_qm)
    
    def jsd_derivatives(self):
        """Compute derivatives of JSD with respect to alpha and beta values."""
        P, Q = self._estimate_densities()
        P = P + self.eps
        Q = Q + self.eps
        M = (P + Q) / 2
        coeff_P = 0.5 * np.log(P / M)
        coeff_Q = 0.5 * np.log(Q / M)
        
        d_alpha = np.zeros(len(self.alpha))
        for i, ai in enumerate(self.alpha):
            dist = (self.bins - ai) / self.h
            kernel_deriv = -self._gaussian_kernel_derivative(dist) / self.h
            d_alpha[i] = np.sum(coeff_P * kernel_deriv) / (len(self.alpha) * self.h)
        
        d_beta = np.zeros(len(self.beta))
        for j, bj in enumerate(self.beta):
            dist = (self.bins - bj) / self.h
            kernel_deriv = -self._gaussian_kernel_derivative(dist) / self.h
            d_beta[j] = np.sum(coeff_Q * kernel_deriv) / (len(self.beta) * self.h)
        
        return d_alpha, d_beta

