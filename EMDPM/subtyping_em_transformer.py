import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from .optimizer_theta_globals import fit_theta_globals
from .optimizer_theta_cluster import fit_theta_cluster
from .optimizer_beta import estimate_beta_for_patient, estimate_beta, beta_loss, beta_loss_jac
from .optimizer_cognitive_regression import fit_linear_cog_regression_multi
from .kernel_jsd import KernelJSD
from .utils import *

class SubtypingEM(BaseEstimator, TransformerMixin):
    """
    EM algorithm for recovering disease progression model parameters
    and patient-specific time shifts from cross-sectional observations.
    """

    def __init__(self, 
                 # [model parameters]
                 max_iter: int = 50,
                 t_max: float = 30,
                 step: float = 0.01,
                 K: np.ndarray = None,
                 rng: np.random.Generator = None,
                 
                # [hyperparameters]
                lambda_f: float = 0.01,
                lambda_cog: float = 0,
                lambda_scalar: float = 0.0,
                lambda_jsd: float = 0.0,  # JSD regularization for beta separation
                 
                 # [initial guesses]
                 initial_f: np.ndarray = None, 
                 initial_assignments: np.ndarray = None,
                 # initial_s
                 # initial_s_K
                 
                 # [iterative fitting parameters]
                 jac_toggle: bool = False,
                 epsilon: float = 1e-2,
                 relative_tolerance: float = 1e-3,
                 
                 # [clustering parmaters]
                 n_subtypes = 2,
                  
                 # [misc options]
                 verbose = 1,
                 ):

        # [model settings]
        self.max_iter = max_iter
        self.t_max = t_max
        self.step = step
        self.K = K
        
        if rng is None:
            self.rng = np.random.default_rng(75)
        else:
            self.rng = rng

        # [hyperparameters]
        self.lambda_f = lambda_f
        self.lambda_cog = lambda_cog
        self.lambda_scalar = lambda_scalar
        self.lambda_jsd = lambda_jsd
        
        # [initial guesses]
        self.initial_f = initial_f
        self.initial_assignments = initial_assignments
        
        # [fitting params]
        self.jac_toggle = jac_toggle
        self.epsilon = epsilon
        self.relative_tolerance = relative_tolerance
        
        # [clustering params]
        self.n_subtypes = n_subtypes
        
        # [misc options]
        self.verbose = verbose
        
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        
        ## data handling
        patient_ids = [p["id"] for p in X]
        n_patients = len(patient_ids)
        n_biomarkers = X[0]["X_obs"].shape[1]
        self.t_span = np.linspace(0.0, self.t_max, int(self.t_max / self.step))

        X_obs_list = []
        dt_list = []
        ids_list = []
        cog_list = []
        initial_beta_list = []

        for i, patient in enumerate(X):
            n = len(patient["dt"])
            X_obs_list.append(patient["X_obs"])
            dt_list.append(patient["dt"])
            ids_list.append(np.full(n, i))
            cog_list.append(ensure_2d_cog(patient["cog"], n)) 
            if "initial_beta" in patient:
                initial_beta_list.append(patient["initial_beta"])

        X_obs = np.vstack(X_obs_list)
        dt    = np.concatenate(dt_list)
        ids   = np.concatenate(ids_list)
        cog   = np.vstack(cog_list)

        # assert
        if not (len(dt) == len(ids) == X_obs.shape[0] == cog.shape[0]):
            raise ValueError(
                f"Stacked shapes disagree: X_obs={X_obs.shape}, dt={dt.shape}, "
                f"ids={ids.shape}, cog={cog.shape}"
            )

        # print(X_obs.shape, dt.shape, cog.shape, ids.shape)

        if initial_beta_list:
            initial_beta = np.array(initial_beta_list)
        else:
            initial_beta = initialize_beta(ids=np.arange(n_patients), beta_range=(0, self.t_max), rng=self.rng)

        max_val = np.max(initial_beta)
        mask = (initial_beta > max_val - 1) & (initial_beta < max_val)
        initial_beta[mask] -= 2
        
        K = self.K

        rng = self.rng
        self.t_span = np.linspace(0, self.t_max, int(self.t_max / self.step)) # I dont know why this is a class variable might be for plotting
        n_biomarkers = np.size(X_obs,1) # rows = observations, columns = biomarkers
        
        n_obs = X_obs.shape[0]
        
        if cog.ndim == 1: # cog.shape = (n_obs, n_cog_features)
            cog = np.atleast_2d(cog)
            cog = cog.T
    
        n_cog_features = cog.shape[1]
        
        ## initialize jacobian logic
        best_lse = np.inf # keep outside loop or else it resets
        
        current_jac = self.jac_toggle
        jacobian_switched = False
        
        ## initialize guesses
        # theta
        initial_x0 = np.zeros(n_biomarkers)
        
        # forcing term, random initialization if None
        if self.initial_f is not None:
            initial_f = self.initial_f
            # Ensure initial_f is 1D (flatten if 2D)
            initial_f = np.ravel(initial_f)
        else:
            initial_f = rng.uniform(0, 0.1, size=n_biomarkers)
            
        initial_s = rng.uniform(0.1, 3, size=n_biomarkers)
        # initial_scalar_K = float(rng.uniform(0.01, 3, size=1))
        initial_scalar_K = float(np.max(X_obs))
        
        initial_theta = np.concatenate([initial_f, initial_s, [initial_scalar_K]])
        
        # cog regression params
        initial_cog_a = np.ones(n_cog_features) # initialize a weight for each type of cog test
        initial_cog_b = 0 # bias term
                
        ## initialize histories
        theta_history = np.zeros((initial_theta.shape[0], self.max_iter + 1)) # extra column added for initial guesses
        beta_history = np.zeros((n_patients, self.max_iter + 1))
        lse_history = np.zeros(self.max_iter + 1)
        cog_regression_history = np.zeros((n_cog_features + 1, self.max_iter + 1))  # + 1 is for intercept
        
        ## Append initial values to histories
        theta_history[:, 0] = initial_theta
        beta_history[:, 0] = initial_beta
        cog_regression_history[:, 0] = np.concatenate([initial_cog_a, [initial_cog_b]])
        
        ## Compute Initial LSE
        X_pred = solve_system(initial_x0, initial_f, K, self.t_span, scalar_K=initial_scalar_K)
        initial_lse = 0.0
        
        for idx, pid in enumerate(np.unique(ids)): # each iter will be like (idx, pid)
            mask = (ids == pid)
            # print(np.size(mask), X_obs.shape, dt.shape, cog.shape)
            X_obs_i = X_obs[mask,:]
            dt_i = dt[mask]
            cog_i = cog[mask,:]
            beta_i = initial_beta[idx]
            
            if current_jac:
                res, _ = beta_loss_jac(beta_i, X_obs_i, dt_i, X_pred, self.t_span, cog_i, initial_cog_a, initial_cog_b, initial_theta, K, self.lambda_cog)
            else:
                res = beta_loss(beta_i, X_obs_i, dt_i, X_pred, self.t_span, cog_i, initial_cog_a, initial_cog_b, initial_theta, self.lambda_cog)
            initial_lse += res
            
        lse_history[0] = initial_lse
        
        # initialize current vars for main loop
        current_beta = initial_beta
        current_s = initial_s
        current_cog_a = initial_cog_a
        current_cog_b = initial_cog_b
        
        # Initialize cluster-level parameters
        # Each cluster has its own f and scalar_K
        cluster_f = []
        cluster_scalar_K = []
        for subtype in range(self.n_subtypes):
            if self.initial_f is not None:
                # Ensure initial_f is 1D before using it
                initial_f_flat = np.ravel(self.initial_f)
                cluster_f.append(initial_f_flat.copy() + rng.uniform(-0.01, 0.01, size=n_biomarkers))
            else:
                cluster_f.append(rng.uniform(0, 0.1, size=n_biomarkers))
            cluster_scalar_K.append(initial_scalar_K * rng.uniform(0.8, 1.2))
        
        # Initialize cluster assignments (use provided or random)
        if self.initial_assignments is not None:
            if len(self.initial_assignments) != n_patients:
                raise ValueError(
                    f"initial_assignments length ({len(self.initial_assignments)}) "
                    f"must match number of patients ({n_patients})"
                )
            if np.any(self.initial_assignments < 0) or np.any(self.initial_assignments >= self.n_subtypes):
                raise ValueError(
                    f"initial_assignments must be in range [0, {self.n_subtypes})"
                )
            assignments = self.initial_assignments.copy()
        else:
            assignments = rng.integers(0, self.n_subtypes, size=n_patients)
        
        # Store assignment history
        assignment_history = np.zeros((n_patients, self.max_iter + 1), dtype=int)
        assignment_history[:, 0] = assignments
        
        ### MAIN LOOP ###
        loop_iter = 0
        
        if self.verbose >= 1:
            pbar = tqdm(total=self.max_iter)
        else:
            # Create a dummy progress bar that does nothing
            class DummyProgressBar:
                def update(self, n=1):
                    pass
            pbar = DummyProgressBar()
            
        while loop_iter < self.max_iter:
            hist_idx = loop_iter + 1
            
            ## STEP 1: GLOBAL LEVEL --> update s (using all patients)
            # Use average f and scalar_K across clusters as reference for global s update
            # For now, use the first cluster's parameters as reference
            ref_f = np.ravel(cluster_f[0])  # Ensure ref_f is 1D
            ref_scalar_K = cluster_scalar_K[0]
            
            current_s = fit_theta_globals(
                X_obs=X_obs, dt_obs=dt, ids=ids, K=K,
                t_span=self.t_span, use_jacobian=current_jac,
                f=ref_f, scalar_K=ref_scalar_K,
                beta_pred=current_beta,
                s_guess=current_s,
                lambda_s=0.0,
                rng=rng
            )
            
            ## STEP 2: RECOMPUTE CLUSTER ASSIGNMENTS (hard assignment)
            # Assign each patient to the cluster that gives lowest reconstruction error
            assignments = self._update_assignments(
                X_obs, dt, ids, cog, current_beta,
                cluster_f, cluster_scalar_K, current_s,
                K, self.t_span, current_cog_a, current_cog_b,
                self.lambda_cog
            )
            assignment_history[:, hist_idx] = assignments
            
            ## STEP 3: CLUSTER LEVEL --> update f[subtype] and scalar_K[subtype] for each cluster
            # TODO: This should be run in parallel
            for subtype in range(self.n_subtypes):
                # get mask for patients in this cluster
                cluster_mask = (assignments == subtype)
                cluster_patient_indices = np.where(cluster_mask)[0]
                
                if len(cluster_patient_indices) == 0:
                    # empty cluster - skip or reinitialize
                    if self.verbose >= 2:
                        print(f"Warning: Cluster {subtype} is empty at iteration {loop_iter}")
                    continue
                
                # Get observations for patients in this cluster
                cluster_patient_mask = np.isin(ids, cluster_patient_indices)
                X_obs_cluster = X_obs[cluster_patient_mask, :]
                dt_cluster = dt[cluster_patient_mask]
                ids_cluster = ids[cluster_patient_mask]
                
                # Map original patient indices to cluster-local indices
                unique_cluster_ids = np.unique(ids_cluster)
                cluster_id_to_local = {orig_id: local_idx for local_idx, orig_id in enumerate(unique_cluster_ids)}
                ids_cluster_local = np.array([cluster_id_to_local[i] for i in ids_cluster])
                beta_cluster = current_beta[cluster_patient_indices]
                
                # Update cluster parameters
                # Ensure f_guess is 1D
                f_guess_flat = np.ravel(cluster_f[subtype])
                f_cluster, scalar_K_cluster = fit_theta_cluster(
                    X_obs=X_obs_cluster, dt_obs=dt_cluster, ids=ids_cluster_local, K=K,
                    t_span=self.t_span, use_jacobian=current_jac,
                    s=current_s,
                    lambda_f=self.lambda_f, lambda_scalar=self.lambda_scalar,
                    beta_pred=beta_cluster,
                    f_guess=f_guess_flat,
                    scalar_K_guess=cluster_scalar_K[subtype],
                    rng=rng
                )
                
                # Ensure f_cluster is 1D before storing
                cluster_f[subtype] = np.ravel(f_cluster)
                cluster_scalar_K[subtype] = scalar_K_cluster
            
            ## STEP 4: SUBJECT LEVEL BETA --> update beta using cluster-level theta
            # Vectorized optimization of all betas simultaneously with JSD included
            current_beta, lse = estimate_beta(
                beta_all=current_beta,
                X_obs=X_obs,
                dt=dt,
                ids=ids,
                cog=cog,
                t_span=self.t_span,
                cluster_f=cluster_f,
                cluster_scalar_K=cluster_scalar_K,
                s=current_s,
                assignments=assignments,
                K=K,
                cog_a=current_cog_a,
                cog_b=current_cog_b,
                lambda_cog=self.lambda_cog,
                lambda_jsd=self.lambda_jsd,
                t_max=self.t_max
            )
            
            beta_history[:, hist_idx] = current_beta
            
            # Store cluster parameters in history (for first cluster as representative)
            # Note: In full implementation, you might want to store all cluster parameters
            representative_theta = np.concatenate([np.ravel(cluster_f[0]), current_s, [cluster_scalar_K[0]]])
            theta_history[:, hist_idx] = representative_theta

            # if self.use_jacobian and lse > best_lse and not jacobian_switched:
            # if self.use_jacobian and best_lse - lse > 1e-3 * best_lse and not jacobian_switched:
            
            delta = best_lse - lse
            
            if (self.jac_toggle == True) and (delta < self.epsilon) and loop_iter > 3:# or lse > best_lse * (1 + self.relative_tolerance)):
                if jacobian_switched == True: # early convergence detected
                    if self.verbose >= 2:
                        print("L-BFGS and Nelder-Mead both failed to improve LSE, exiting early due to convergence")
                    current_cog_a, current_cog_b = fit_linear_cog_regression_multi(cog, dt, current_beta, ids)
                    cog_regression_history[:, hist_idx] = np.concatenate([current_cog_a, [current_cog_b]])
                    lse_history[hist_idx] = lse
                    break 
                
                current_jac = False
                if self.verbose >= 2:
                    print(f"warning: toggling to jac {current_jac}; due to increase or convergence in LSE at iteration {loop_iter}.")
                jacobian_switched = True
                continue
            # if best_lse - lse > 1e-3 * best_lse
                # continue # skip storing values for current iteration. if True --> DONT UPDATE and RETRY
                
            ## update accepted
            jacobian_switched = False
            if self.jac_toggle == True:
                current_jac = True
            
            best_lse = min(best_lse, lse)
            # TODO: Get idx of best lse
            lse_history[hist_idx] = lse
            
            current_cog_a, current_cog_b = fit_linear_cog_regression_multi(cog, dt, current_beta, ids)
            cog_regression_history[:, hist_idx] = np.concatenate([current_cog_a, [current_cog_b]])
            
            loop_iter += 1
            pbar.update(1)
            
            
        self.theta_history = theta_history[:, 0:hist_idx+1]
        self.beta_history = beta_history[:, 0:hist_idx+1]
        self.lse_history = lse_history[0:hist_idx+1]
        self.cog_regression_history = cog_regression_history[:, 0:hist_idx+1]
        self.assignment_history = assignment_history[:, 0:hist_idx+1]
        
        # Store final cluster parameters
        self.cluster_f = cluster_f
        self.cluster_scalar_K = cluster_scalar_K
        self.final_s = current_s
        self.final_assignments = assignments
        self.subtype_mapping = None  # Will be set if compute_subtype_mapping is called
        
        # Match EM labels to true labels if available
        # Check if X contains true subtype labels
        if X is not None and len(X) > 0 and "subtype_true" in X[0]:
            true_labels = np.array([p.get("subtype_true", -1) for p in X])
            if np.all(true_labels >= 0):  # Only match if all have valid true labels
                try:
                    self.final_assignments_matched = match_labels_best_overlap(
                        assignments, true_labels
                    )
                    self.label_mapping_applied = True
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"Warning: Could not match labels: {e}")
                    self.final_assignments_matched = assignments.copy()
                    self.label_mapping_applied = False
            else:
                self.final_assignments_matched = assignments.copy()
                self.label_mapping_applied = False
        else:
            self.final_assignments_matched = assignments.copy()
            self.label_mapping_applied = False
        
        # Store representative theta (first cluster)
        self.theta = np.concatenate([np.ravel(cluster_f[0]), current_s, [cluster_scalar_K[0]]])
        self.cog_a = self.cog_regression_history[:-1, -1]
        self.cog_b = self.cog_regression_history[-1, -1]
        
        self.final_f = np.ravel(cluster_f[0]).copy()  # Representative f (ensure 1D)
        self.scalar_K_ = cluster_scalar_K[0]  # Representative scalar_K

        # For transform, use first cluster as default
        f = np.ravel(cluster_f[0])  # Ensure 1D
        scalar_K = cluster_scalar_K[0]
        self.X_pred = solve_system(np.zeros(n_biomarkers), f, self.K, self.t_span, scalar_K)
        
        return self
    
    def _update_assignments(self, X_obs, dt, ids, cog, beta, cluster_f, cluster_scalar_K, s,
                            K, t_span, cog_a, cog_b, lambda_cog):
        """
        Update cluster assignments using hard assignment based on reconstruction error.
        
        For each patient, compute reconstruction error with each cluster's parameters
        and assign to the cluster with lowest error.
        """
        n_patients = len(np.unique(ids))
        n_subtypes = len(cluster_f)
        assignments = np.zeros(n_patients, dtype=int)
        
        unique_ids = np.unique(ids)
        
        for idx, patient_id in enumerate(unique_ids):
            mask = (ids == patient_id)
            X_obs_i = X_obs[mask, :]
            dt_i = dt[mask]
            cog_i = cog[mask, :]
            beta_i = beta[idx]
            
            best_error = np.inf
            best_subtype = 0
            
            # Try each cluster and find the one with lowest reconstruction error
            for subtype in range(n_subtypes):
                f_cluster = np.ravel(cluster_f[subtype])  # Ensure 1D
                scalar_K_cluster = cluster_scalar_K[subtype]
                theta_cluster = np.concatenate([f_cluster, s, [scalar_K_cluster]])
                X_pred_cluster = solve_system(np.zeros(X_obs_i.shape[1]), f_cluster, K, t_span, scalar_K_cluster)
                
                # Compute reconstruction error
                error = beta_loss(
                    beta_i, X_obs_i, dt_i, X_pred_cluster, t_span,
                    cog_i, cog_a, cog_b, theta_cluster, lambda_cog
                )
                
                if error < best_error:
                    best_error = error
                    best_subtype = subtype
            
            assignments[idx] = best_subtype
        
        return assignments


    def _update_assignments_likelihood(self, X_obs, dt, ids, cog, beta, cluster_f, 
                                   cluster_scalar_K, s, K, t_span, cog_a, cog_b, 
                                   lambda_cog, error_scale=1.0):
        """Probabilistic assignments assuming Gaussian error distribution."""
        n_patients = len(np.unique(ids))
        n_subtypes = len(cluster_f)
        probabilities = np.zeros((n_patients, n_subtypes))
        
        unique_ids = np.unique(ids)
        
        for idx, patient_id in enumerate(unique_ids):
            mask = (ids == patient_id)
            X_obs_i = X_obs[mask, :]
            dt_i = dt[mask]
            cog_i = cog[mask, :]
            beta_i = beta[idx]
            
            log_likelihoods = np.zeros(n_subtypes)
            
            for subtype in range(n_subtypes):
                f_cluster = np.ravel(cluster_f[subtype])
                scalar_K_cluster = cluster_scalar_K[subtype]
                theta_cluster = np.concatenate([f_cluster, s, [scalar_K_cluster]])
                X_pred_cluster = solve_system(np.zeros(X_obs_i.shape[1]), f_cluster, K, t_span, scalar_K_cluster)
                
                error = beta_loss(
                    beta_i, X_obs_i, dt_i, X_pred_cluster, t_span,
                    cog_i, cog_a, cog_b, theta_cluster, lambda_cog
                )
                
                # Gaussian log-likelihood: -0.5 * (error/scale)^2
                log_likelihoods[subtype] = -0.5 * (error / error_scale) ** 2
            
            # Convert to probabilities (with numerical stability)
            log_likelihoods -= np.max(log_likelihoods)
            probabilities[idx] = np.exp(log_likelihoods)
            probabilities[idx] /= np.sum(probabilities[idx])
        
            assignments = np.argmax(probabilities, axis=1)
            return assignments, probabilities
    
    def _optimize_jsd_redistribution(self, beta, assignments, t_max, iteration=None):
        """
        Optimize JSD redistribution using gradient descent (multiple steps).
        This is called AFTER all betas are optimized to redistribute them based on JSD.
        
        Much faster than computing JSD thousands of times during individual beta optimization.
        Does actual optimization (multiple gradient steps) rather than just one step.
        
        Parameters
        ----------
        beta : np.ndarray
            Current beta values for all patients (already optimized for reconstruction)
        assignments : np.ndarray
            Subtype assignments for each patient
        t_max : float
            Maximum time value (for bounds)
        iteration : int, optional
            Current iteration number for verbose output
        
        Returns
        -------
        np.ndarray
            Redistributed beta values
        """
        if self.n_subtypes != 2:
            return beta
        
        # Extract betas for each subtype
        subtype_0_betas = beta[assignments == 0]
        subtype_1_betas = beta[assignments == 1]
        
        if len(subtype_0_betas) == 0 or len(subtype_1_betas) == 0:
            return beta
        
        idx_subtype_0 = np.where(assignments == 0)[0]
        idx_subtype_1 = np.where(assignments == 1)[0]
        
        beta_optimized = beta.copy()
        jsd_before = None
        
        # Do multiple gradient descent steps to optimize JSD
        n_jsd_steps = max(1, int(self.lambda_jsd * 0.1)) 
        
        for step in range(n_jsd_steps):
            # Extract current betas for each subtype
            subtype_0_current = beta_optimized[idx_subtype_0]
            subtype_1_current = beta_optimized[idx_subtype_1]
            
            # Compute JSD and derivatives
            jsd_calc = KernelJSD(
                alpha=subtype_0_current,
                beta=subtype_1_current,
                value_range=(0, t_max)
            )
            
            if step == 0:
                jsd_before = jsd_calc.jsd()
            
            d_alpha, d_beta = jsd_calc.jsd_derivatives()
            
            # Adaptive step size - normalize gradients to prevent overshooting
            grad_norm_alpha = np.linalg.norm(d_alpha) if len(d_alpha) > 0 else 1.0
            grad_norm_beta = np.linalg.norm(d_beta) if len(d_beta) > 0 else 1.0
            
            # Step size: smaller for later steps (fine-tuning)
            step_size = self.lambda_jsd * 0.01 * (1.0 / (step + 1))
                        
            # Apply gradient step to maximize JSD
            beta_optimized[idx_subtype_0] += step_size * d_alpha
            beta_optimized[idx_subtype_1] += step_size * d_beta
            
            # Clip to valid range
            beta_optimized = np.clip(beta_optimized, 0, t_max)
        
        # Diagnostic output
        if self.verbose >= 2 and iteration is not None and iteration % 10 == 0:
            jsd_after_calc = KernelJSD(
                alpha=beta_optimized[idx_subtype_0],
                beta=beta_optimized[idx_subtype_1],
                value_range=(0, t_max)
            )
            jsd_after = jsd_after_calc.jsd()
            beta_change = np.mean(np.abs(beta_optimized - beta))
            print(f"  Iter {iteration}: JSD opt steps={n_jsd_steps}, "
                  f"JSD {jsd_before:.6f} -> {jsd_after:.6f}, "
                  f"mean beta change={beta_change:.6f}")
        
        return beta_optimized
    
    def compute_subtype_mapping(self, true_f_list, verbose=True):
        """
        Compute subtype mapping based on fitted f vs true f parameters.
        
        Parameters
        ----------
        true_f_list : Sequence[np.ndarray]
            List of true f arrays, one per subtype.
        verbose : bool
            Whether to print the mapping.
        """
        from .utils import get_subtype_mapping_from_f
        self.subtype_mapping = get_subtype_mapping_from_f(self.cluster_f, true_f_list)
        if verbose:
            print(f"\nSubtype mapping (fitted -> true): {self.subtype_mapping}")
            for fitted_subtype in range(self.n_subtypes):
                print(f"  Fitted subtype {fitted_subtype} -> True subtype {self.subtype_mapping[fitted_subtype]}")
        return self.subtype_mapping
    
    def transform(self, X: list[dict]) -> np.ndarray:
        """
        Estimate beta values for a list of patient dicts.
        """
        beta_vals = []
        for p in tqdm(X, desc="Estimating beta values"):
            cog_i = p["cog"]
            if cog_i.ndim == 1:
                cog_i = cog_i.reshape(-1, 1)

            beta_i = estimate_beta_for_patient(
                beta_i=0.0,
                X_obs_i=p["X_obs"],
                dt_i=p["dt"],
                X_pred=self.X_pred,
                t_span=self.t_span,
                cog_i=cog_i,
                cog_a=self.cog_a,
                cog_b=self.cog_b,
                theta=self.theta,
                K=self.K,
                lambda_cog=self.lambda_cog,
                use_jacobian=self.jac_toggle,
                t_max=self.t_max
            )
            beta_vals.append(beta_i)
        self.beta_val = beta_vals
        return np.array(beta_vals)


    def score(self, X: dict, y=None) -> float:
        """
        Computes timeshifts of validation set using transform,
        evaluates difference between predicted model and obs
        """
        beta_val = self.transform(X)
        lse = self._compute_val_score(X, beta_val)
        return -lse
    
    def _compute_val_score(self, X: list[dict], beta: np.ndarray) -> float:
        n_biomarkers = X[0]["X_obs"].shape[1]
        f = self.theta[:n_biomarkers]
        s = self.theta[n_biomarkers:2 * n_biomarkers]
        scalar_K = self.theta[-1]

        X_pred = solve_system(np.zeros(n_biomarkers), f, self.K, self.t_span, scalar_K)

        lse = 0.0
        for i, p in enumerate(X):
            dt_i = p["dt"]
            X_obs_i = p["X_obs"]
            beta_i = beta[i]
            time_points = beta_i + dt_i

            X_interp = np.vstack([
                np.interp(time_points, self.t_span, X_pred[b]) * s[b]
                for b in range(n_biomarkers)
            ]).T

            lse += np.sum((X_obs_i - X_interp) ** 2)
        return lse
    
    def _ensure_2d_cog(c, n_rows_expected: int) -> np.ndarray:
        c = np.asarray(c)
        # (n_obs,) -> (n_obs, 1)
        if c.ndim == 1:
            c = c.reshape(-1, 1)

        # (1, n_obs) -> (n_obs, 1)
        if c.ndim == 2 and c.shape[0] == 1 and c.shape[1] == n_rows_expected:
            c = c.T

        # Final check: first dim must match number of observations for this patient
        if c.ndim != 2 or c.shape[0] != n_rows_expected:
            raise ValueError(
                f"cog shape mismatch; expected first dim {n_rows_expected}, got {c.shape}"
            )
        return c


def fit_subtyping_em_with_assignments(
    X: list,
    initial_assignments: np.ndarray,
    em_kwargs: dict,
    run_index: int = 0,
    seed_offset: int = 0
):
    """
    Fit a SubtypingEM model with a specific initial assignment.
    Designed to be called in parallel.
    
    Parameters
    ----------
    X : list
        List of patient dictionaries
    initial_assignments : np.ndarray
        Initial cluster assignments for each patient
    em_kwargs : dict
        Keyword arguments to pass to SubtypingEM constructor
    run_index : int
        Index of this run (for RNG seeding)
    seed_offset : int
        Base seed offset for RNG
    
    Returns
    -------
    dict
        Dictionary containing the fitted model and results
    """
    rng = np.random.default_rng(75 + seed_offset + run_index)
    
    # Create a copy of em_kwargs and add rng and initial_assignments
    kwargs = em_kwargs.copy()
    kwargs['rng'] = rng
    kwargs['initial_assignments'] = initial_assignments
    
    em = SubtypingEM(**kwargs)
    em.fit(X=X, y=None)
    
    result = {
        'run_index': run_index,
        'model': em,
        'beta_history': em.beta_history,
        'lse_history': em.lse_history,
        'assignment_history': em.assignment_history,
        'final_assignments': em.final_assignments,
        'initial_assignments': initial_assignments,
        'final_lse': em.lse_history[-1] if len(em.lse_history) > 0 else np.inf,
    }
    
    return result


def run_multiple_initializations_parallel(
    X: list,
    n_initializations: int,
    em_kwargs: dict,
    n_jobs: int = -1,
    prefer: str = "processes",
    seed_offset: int = 0,
    rng: np.random.Generator = None
):
    """
    Run multiple SubtypingEM fits with different random initial assignments in parallel.
    
    Parameters
    ----------
    X : list
        List of patient dictionaries
    n_initializations : int
        Number of different initializations to try
    em_kwargs : dict
        Keyword arguments to pass to SubtypingEM constructor
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    prefer : str
        Preferred backend: "processes" or "threads"
    seed_offset : int
        Base seed offset for RNG
    rng : np.random.Generator, optional
        Random number generator for creating initial assignments
        
    Returns
    -------
    list
        List of result dictionaries, one per initialization
    """
    from joblib import Parallel, delayed
    
    n_patients = len(X)
    n_subtypes = em_kwargs.get('n_subtypes', 2)
    
    if rng is None:
        rng = np.random.default_rng(75 + seed_offset)
    
    # Generate random initial assignments for each run
    initial_assignments_list = [
        rng.integers(0, n_subtypes, size=n_patients)
        for _ in range(n_initializations)
    ]
    
    # Create jobs
    jobs = []
    for idx, assignments in enumerate(initial_assignments_list):
        jobs.append(delayed(fit_subtyping_em_with_assignments)(
            X, assignments, em_kwargs, run_index=idx, seed_offset=seed_offset
        ))
    
    # Run in parallel
    results = Parallel(n_jobs=n_jobs, prefer=prefer)(jobs)
    
    # Find best result by final LSE
    best_idx = np.argmin([r['final_lse'] for r in results])
    
    return results, best_idx

