import numpy as np
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
                 lamda: float = 0.01,
                 lambda_cog: float = 0,
                 rng: np.random.Generator = None
                 ):

        # [fitter settings]
        self.num_iterations = num_iterations
        self.t_max = t_max
        self.step = step
        self.rng = np.random.default_rng(75)
        
        # [hyperparameters]
        self.lamda = lamda # TODO: consider refactoring to like "lasso_penalty" or "lambda_connectiviy"
        self.lambda_cog = lambda_cog
        
        ## [jacobian switching logic]
        self.use_jacobian = use_jacobian
        
    def fit(self,
            X: np.ndarray,
            dt: np.ndarray,
            ids: np.ndarray,
            cog: np.ndarray,
            K: np.ndarray,
            y: np.ndarray = None):
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
        cog = np.atleast_2d(cog) # cast cog into 2D array size tuple needs two elements for counting cols
        n_cog_features = cog.shape[0]
        
        ## initialize jacobian logic
        best_lse = np.inf # keep outside loop or else it resets
        jacobian_switched = False
        
        ## initialize guesses
        # theta
        initial_x0 = np.zeros(n_biomarkers)
        initial_f = rng.uniform(0, 0.2, size=n_biomarkers)
        initial_s = rng.uniform(0.1, 3, size=n_biomarkers)
        initial_scalar_K = float(rng.uniform(0.01, 3, size=1))
        initial_theta = np.concatenate([initial_f, initial_s, [initial_scalar_K]])
        
        # beta
        initial_beta = initialize_beta(ids=ids, beta_range=(0, self.t_max), rng=rng)
        
        # cog regression params
        initial_cog_a = np.ones(n_cog_features) # initialize a weight for each type of cog test
        initial_cog_b = 0 # bias term
        
        #initial_cog_a = rng.uniform(1, 5, n_cog_features) # initialize a weight for each type of cog test
        #initial_cog_b = float(rng.uniform(0, 10)) # bias term
        
        ## move these later towards the end for a summary:
        print("initial conditions:")
        print(f"n_patients: {n_patients}, n_obs: {n_obs}")
        print(f"initial f: {initial_f}")
        print(f"initial s: {initial_s}")
        print(f"initial scalar K: {initial_scalar_K}")
        print(f"initial beta: {initial_beta.shape}")
        
        ## initialize histories
        theta_history = np.zeros((initial_theta.shape[0], self.num_iterations + 1)) # extra column added for initial guesses
        beta_history = np.zeros((n_patients, self.num_iterations + 1))
        lse_history = np.zeros((n_patients, self.num_iterations + 1))
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
            X_obs_i, dt_i, cog_i = X[mask], dt[mask], cog.T[mask]
            beta_i = initial_beta[idx]
            
            if self.use_jacobian:
                res, _ = beta_loss_jac(beta_i, X_obs_i, dt_i, X_pred, self.t_span, cog_i, initial_cog_a, initial_cog_b, initial_theta, K, self.lambda_cog)
            else:
                res = beta_loss(beta_i, X_obs_i, dt_i, X_pred, self.t_span, cog_i, initial_cog_a, initial_cog_b, self.lambda_cog)
            initial_lse += res
            
        #print(f"LSE Prepend complete! LSE: {initial_lse}")
        lse_history[0] = initial_lse
        
        ## initialize current vars for main loop
        current_beta = initial_beta
        current_f = initial_f
        current_s = initial_s
        current_scalar_K = initial_scalar_K
        current_cog_a = initial_cog_a
        current_cog_b = initial_cog_b
        
        ## MAIN LOOP
        loop_iter = 0
        
        pbar = tqdm(total=self.num_iterations)
        while loop_iter < self.num_iterations:
            hist_idx = loop_iter + 1
            current_theta = fit_theta(X_obs=X, dt_obs=dt, ids=ids, K=K,
                                             t_span=self.t_span, step=self.step,
                                             use_jacobian=self.use_jacobian, lamda=self.lamda,
                                             beta_pred=current_beta, f_guess=current_f, s_guess=initial_s,
                                             scalar_K_guess=current_scalar_K, rng=rng
                                      )
            
            current_x0, current_f, current_s, current_scalar_K = current_theta
            X_pred = solve_system(current_x0, current_f, K, self.t_span, current_scalar_K)
            # print("Breakpoint 7: ", X_pred.shape, X_pred.dtype)
            theta_history[:,hist_idx] = np.concatenate((current_f, current_s, [current_scalar_K]))
            lse = 0.0
            
            # TODO: Attempt parallel beta computation
            ### beta comuputaiton
            for idx, patient_id in enumerate(np.unique(ids)):
                mask = (ids == patient_id)
                x_obs_i = X[mask]  # (n_obs_i, n_biomarkers)
                dt_i = dt[mask]    # (n_obs_i,)
                cog_i = cog.T[mask]  # (n_obs_i, n_cog_features)
                beta_i = current_beta[idx]
            
                #print("breakpoint 6: ", x_obs_i.shape, x_obs_i.dtype,dt_i.shape, dt_i.dtype, cog_i.shape, cog_i.dtype, beta_i)  
                #print("breakpoint 8: ", current_cog_a.shape)
            
                beta_i = estimate_beta_for_patient(beta_i=beta_i, X_obs_i=x_obs_i, dt_i=dt_i,
                                                   X_pred=X_pred, t_span=self.t_span,
                                                   cog_i = cog_i, cog_a = current_cog_a, cog_b = current_cog_b,
                                                   theta = current_theta, K=K, lambda_cog = self.lambda_cog,
                                                   use_jacobian = self.use_jacobian, t_max = self.t_max
                                                   )
                
                beta_history[idx, hist_idx] = beta_i
                
                if self.use_jacobian:
                    res, _ = beta_loss_jac(beta_i, X_obs_i, dt_i, X_pred, self.t_span, cog_i, current_cog_a, current_cog_b, initial_theta, K, self.lambda_cog)
                else:
                    res = beta_loss(beta_i, X_obs_i, dt_i, X_pred, self.t_span, cog_i, current_cog_a, current_cog_b, self.lambda_cog)
                lse += res

            if self.use_jacobian and lse > best_lse and not jacobian_switched:
                print(f"warning: jacobian toggled off due to increase in LSE at iteration {loop_iter}.")
                self.use_jacobian = False
                jacobian_switched = True
                
                # continue # skip storing values for current iteration. if True --> DONT UPDATE and RETRY
                
            ## update accepted
            best_lse = min(best_lse, lse)
            # TODO: Get idk of best lse
            lse_history[hist_idx] = lse
            
            #current_cog_a, current_cog_b = fit_linear_cog_regression_multi(cog, dt, current_beta)
            #cog_regression_history[:, hist_idx] = np.concatenate([current_cog_a, [current_cog_b]])
            
            loop_iter += 1
            pbar.update(1)
            
        final_theta = theta_history[:,-1]

        print("\nSUMMARY:")
        print("initial theta: ", initial_theta)
        print("final theta: ", final_theta)
        
        self.theta_history = theta_history
        self.beta_history = beta_history
        self.lse_history = lse_history
        self.cog_regression_history = cog_regression_history

        return self