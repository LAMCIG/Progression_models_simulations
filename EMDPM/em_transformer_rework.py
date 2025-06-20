import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from .optimizer_theta import fit_theta
from .optimizer_beta import estimate_beta_for_patient, beta_loss, beta_loss_jac
from .optimizer_cognitive_regression import fit_linear_cog_regression_multi
from .utils import solve_system, initialize_beta

class EM(BaseEstimator, TransformerMixin):
    """
    EM algorithm for recovering disease progression model parameters
    and patient-specific time shifts from cross-sectional observations.
    """

    def __init__(self, 
                 num_iterations: int = 50,
                 t_max: float = 12,
                 step: float = 0.01,
                 use_jacobian: bool = False,
                 lambda_f: float = 0.01,
                 lambda_cog: float = 0,
                 lambda_scalar: float = 0.0,
                 rng: np.random.Generator = None
                 ):

        # [fitter settings]
        self.num_iterations = num_iterations
        self.t_max = t_max
        self.step = step
        self.rng = np.random.default_rng(75)
        
        # [hyperparameters]
        self.lambda_f = lambda_f # TODO: consider refactoring to like "lasso_penalty" or "lambda_connectiviy"
        self.lambda_cog = lambda_cog
        self.lambda_scalar = lambda_scalar
        
        ## [jacobian switching logic]
        self.use_jacobian = use_jacobian
        
    def fit(self,
            X: np.ndarray,
            dt: np.ndarray,
            ids: np.ndarray,
            cog: np.ndarray,
            K: np.ndarray,
            initial_beta: np.ndarray = None,
            save_path: str = None):
        """
        Run the EM loop to estimate theta (ODE params) and beta (patient shifts).

        Parameters
        ----------

        Returns
        -------
        self
        """
        rng = self.rng
        self.t_span = np.linspace(0, self.t_max, int(self.t_max / self.step)) # I dont know why this is a class variable might be for plotting
        n_biomarkers = np.size(X,1) # rows = observations, columns = biomarkers
        n_patients = len(np.unique(ids))
        n_obs = X.shape[0]
        
        if cog.ndim == 1: # cog.shape = (n_obs, n_cog_features)
            cog = np.atleast_2d(cog)
            cog = cog.T
    
        n_cog_features = cog.shape[1]
        
        ## initialize jacobian logic
        best_lse = np.inf # keep outside loop or else it resets
        jacobian_toggle = self.use_jacobian
        
        ## initialize guesses
        # theta
        initial_x0 = np.zeros(n_biomarkers)
        initial_f = rng.uniform(0, 0.1, size=n_biomarkers)
        # initial_s = rng.uniform(0.1, 3, size=n_biomarkers)
        initial_s = np.ones(n_biomarkers)*-1
        initial_s0 = np.ones(n_biomarkers) * np.max(X, axis=0)

        initial_scalar_K = np.max(X)
        initial_theta = np.concatenate([initial_f, initial_s, initial_s0, [initial_scalar_K]])
        
        # beta
        if initial_beta is None:
            initial_beta = initialize_beta(ids=ids, beta_range=(0, self.t_max), rng=rng)
            
        # cog regression params
        initial_cog_a = np.ones(n_cog_features) # initialize a weight for each type of cog test
        initial_cog_b = 0 # bias term
        
        ## initialize histories
        theta_history = np.zeros((initial_theta.shape[0], self.num_iterations + 1)) # extra column added for initial guesses
        beta_history = np.zeros((n_patients, self.num_iterations + 1))
        lse_history = np.zeros(self.num_iterations + 1)
        cog_regression_history = np.zeros((n_cog_features + 1, self.num_iterations + 1))  # + 1 is for intercept
        
        ## Append initial values to histories
        theta_history[:, 0] = initial_theta
        beta_history[:, 0] = initial_beta
        cog_regression_history[:, 0] = np.concatenate([initial_cog_a, [initial_cog_b]])
        
        ## Compute Initial LSE
        X_pred = solve_system(initial_x0, initial_f, K, self.t_span, scalar_K=initial_scalar_K)
        #print(X_pred.shape)

        initial_lse = 0.0
        
        for idx, pid in enumerate(np.unique(ids)): # each iter will be like (idx, pid)
            mask = (ids == pid)
            # print(np.size(mask), X.shape, dt.shape, cog.shape)
            X_obs_i, dt_i, cog_i = X[mask,:], dt[mask], cog[mask,:]
            beta_i = initial_beta[idx]
            
            if self.use_jacobian:
                res, _ = beta_loss_jac(beta_i, X_obs_i, dt_i, X_pred, self.t_span, cog_i, initial_cog_a, initial_cog_b, initial_theta, K, self.lambda_cog)
            else:
                res = beta_loss(beta_i, X_obs_i, dt_i, X_pred, self.t_span, cog_i, initial_cog_a, initial_cog_b, initial_theta, self.lambda_cog)
            initial_lse += res
            
        lse_history[0] = initial_lse
        
        # initialize current vars for main loop
        current_beta = initial_beta
        current_f = initial_f
        current_s = initial_s
        current_scalar_K = initial_scalar_K
        current_cog_a = initial_cog_a
        current_cog_b = initial_cog_b
        current_s0 = initial_s0
        
        print(f"prepend complete")
        ### MAIN LOOP ###
        loop_iter = 0
        
        pbar = tqdm(total=self.num_iterations)
        while loop_iter < self.num_iterations:
            hist_idx = loop_iter + 1
            current_theta = fit_theta(X_obs=X, dt_obs=dt, ids=ids, K=K,
                                             t_span=self.t_span, use_jacobian=jacobian_toggle,
                                             lambda_f=self.lambda_f, lambda_scalar=self.lambda_scalar,
                                             beta_pred=current_beta, f_guess=current_f, s0_guess=current_s0,
                                             s_guess=current_s, scalar_K_guess=current_scalar_K, rng=rng
                                      )
            
            current_x0, current_f, current_s, current_s0, current_scalar_K = current_theta
            current_theta = np.concatenate((current_f, current_s, current_s0,[current_scalar_K]))
            X_pred = solve_system(current_x0, current_f, K, self.t_span, current_scalar_K)
            theta_history[:,hist_idx] = np.concatenate((current_f, current_s, current_s0, [current_scalar_K]))
            lse = 0.0
            
            # TODO: Attempt parallel beta computation
            ### beta comuputaiton
            for idx, patient_id in enumerate(np.unique(ids)):
                mask = (ids == patient_id)
                X_obs_i = X[mask,:]  # (n_obs_i, n_biomarkers)
                dt_i = dt[mask]    # (n_obs_i,)
                cog_i = cog[mask,:]  # (n_obs_i, n_cog_features)
                beta_i = current_beta[idx]            
                beta_i = estimate_beta_for_patient(beta_i=beta_i, X_obs_i=X_obs_i, dt_i=dt_i,
                                                   X_pred=X_pred, t_span=self.t_span,
                                                   cog_i = cog_i, cog_a = current_cog_a, cog_b = current_cog_b,
                                                   theta = current_theta, K=K, lambda_cog = self.lambda_cog,
                                                   use_jacobian = True, t_max = self.t_max
                                                   )
       
                if self.use_jacobian == True:
                    res, _ = beta_loss_jac(beta_i, X_obs_i, dt_i, X_pred, self.t_span, cog_i, current_cog_a, current_cog_b, current_theta, K, self.lambda_cog)
                else:
                    res = beta_loss(beta_i, X_obs_i, dt_i, X_pred, self.t_span, cog_i, current_cog_a, current_cog_b, current_theta, self.lambda_cog)
                lse += res
                
                current_beta[idx] = beta_i 
            beta_history[:,hist_idx] = current_beta

            #if self.use_jacobian and lse > best_lse and not jacobian_switched:
            if lse > best_lse - 1e-3 * best_lse:
                if self.use_jacobian:
                    jacobian_toggle = not jacobian_toggle
                    print(f"jacobian toggle: {jacobian_toggle} due to increase or convergence in LSE at iteration {loop_iter}.")
                continue
            
            ## update accepted
            best_lse = min(best_lse, lse)
            # TODO: Get idk of best lse
            lse_history[hist_idx] = lse
            
            current_cog_a, current_cog_b = fit_linear_cog_regression_multi(cog, dt, current_beta, ids)
            cog_regression_history[:, hist_idx] = np.concatenate([current_cog_a, [current_cog_b]])
            
            loop_iter += 1
            pbar.update(1)
            
        #final_theta = theta_history[:,-1]
        
        # print("initial conditions:")
        # print(f"initial f: {initial_f}")
        # print(f"initial s: {initial_s}")
        # print(f"initial scalar K: {initial_scalar_K}")
        # print(f"initial beta: {initial_beta}")
        
        # print("current conditions:")
        # print(f"current f: {current_f}")
        # print(f"current s: {current_s}")
        # print(f"current scalar K: {current_scalar_K}")
        # print(f"current beta: {current_beta}")
        
        self.theta_history = theta_history
        self.beta_history = beta_history
        self.lse_history = lse_history
        self.cog_regression_history = cog_regression_history

        if save_path is not None:
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)

            # save as npz archive
            np.savez_compressed(save_path,
                                theta_history=self.theta_history,
                                beta_history=self.beta_history,
                                lse_history=self.lse_history,
                                cog_regression_history=self.cog_regression_history)
            print(f"Saved histories to {save_path}.npz")
        
        
        return self