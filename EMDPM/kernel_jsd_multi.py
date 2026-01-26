import numpy as np


class KernelJSDMulti:
    """
    Kernel-based Jensen-Shannon Divergence for comparing beta distributions.
    Used to encourage similar beta distributions between subtypes by minimizing
    JSD (for N subtypes).

    h = bandwidth
    """

    def __init__(self, distributions_list, value_range=(0, 1), bandwidth=None, n_bins=None):
        self.distributions = [np.asarray(d) for d in distributions_list]
        self.n_distributions = len(self.distributions)
        self.range = value_range
        
        # Auto-determine number of bins based on range if not specified
        if n_bins is None:
            range_size = value_range[1] - value_range[0]
            self.n_bins = max(20, int(range_size * 2))  # At least 20 bins, or 2 per unit
        else:
            self.n_bins = n_bins
        self.bins = np.linspace(value_range[0], value_range[1], self.n_bins)
        
        if bandwidth is None:
            combined = np.concatenate(self.distributions)
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
        densities = []
        for dist in self.distributions:
            dist_scaled = (self.bins[:, None] - dist[None, :]) / self.h
            P = np.mean(self._gaussian_kernel(dist_scaled), axis=1) / self.h
            P = P / (P.sum() + self.eps)
            densities.append(P)
        return densities
    
    def jsd(self):
        """Compute Jensen-Shannon Divergence for N distributions.
        """
        densities = self._estimate_densities()
        densities = [P + self.eps for P in densities] # eps for numerical stability
        

        M = np.mean(densities, axis=0)
        M = M + self.eps # in case 0
        
        jsd = 0.0
        for P in densities:
            kl = np.sum(P * np.log(P / M))
            jsd += kl
        jsd /= self.n_distributions
        
        return jsd
    
    def jsd_derivatives(self):
        """Compute derivatives of JSD with respect to each distribution's values.
        Returns list of gradient arrays, one per distribution.
        """
        densities = self._estimate_densities()
        # Add epsilon for numerical stability
        densities = [P + self.eps for P in densities]
        
        # compute mixture distribution M = (1/N) * sum(P_i)
        M = np.mean(densities, axis=0)
        M = M + self.eps
        
        # compute coefficients for each distribution: (1/N) * log(P_i / M)
        coeffs = []
        for P in densities:
            coeff = (1.0 / self.n_distributions) * np.log(P / M)
            coeffs.append(coeff)
        
        # Compute gradients for each distribution
        gradients = []
        for dist_idx, dist in enumerate(self.distributions):
            d_dist = np.zeros(len(dist))
            coeff = coeffs[dist_idx]
            
            for i, val in enumerate(dist):
                dist_scaled = (self.bins - val) / self.h
                kernel_deriv = -self._gaussian_kernel_derivative(dist_scaled) / self.h
                d_dist[i] = np.sum(coeff * kernel_deriv) / (len(dist) * self.h)
            
            gradients.append(d_dist)
        
        return gradients

