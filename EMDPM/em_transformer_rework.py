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
                 max_iter: int = 50,
                 t_max: float = 12,
                 step: float = 0.01,
                 jac_toggle: bool = False,
                 lambda_f: float = 0.01,
                 lambda_cog: float = 0,
                 lambda_scalar: float = 0.0,
                 rng: np.random.Generator = None, 
                 K: np.ndarray = None,
                 ):

        # [model settings]
        self.max_iter = max_iter
        self.t_max = t_max
        self.step = step
        
        if rng is None:
            self.rng = np.random.default_rng(75)
        else:
            self.rng = rng

        # [hyperparameters]
        self.lambda_f = lambda_f
        self.lambda_cog = lambda_cog
        self.lambda_scalar = lambda_scalar
        
        # [jacobian switching logic]
        self.jac_toggle = jac_toggle
        
        self.K = K
        
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        import time
        
        # 2025.6.22 - will temporarily assign class vars to regular vars, TODO: refactor class vars
        # ids = self.ids
        # dt = self.dt
        # cog = self.cog
        # K = self.K
        # save_path = self.save_path
        
        # X_obs = X["X_obs"] # 2025.23.6 - I will refactor X in this function to X_obs later -DS
        # dt = X["dt"]
        # ids = X["ids"]
        # cog = X["cog"]
        
        unique_ids = [p["id"] for p in X]
        id_to_index = {pid: i for i, pid in enumerate(unique_ids)}

        X_obs_list = []
        dt_list = []
        ids_list = []
        cog_list = []
        initial_beta_list = []

        for i, patient in enumerate(X):
            n = len(patient["dt"])
            X_obs_list.append(patient["X_obs"])
            dt_list.append(patient["dt"])
            cog_list.append(patient["cog"])
            ids_list.append(np.full(n, i))  # integer index for patient
            if "initial_beta" in patient:
                initial_beta_list.append(patient["initial_beta"])

        X_obs = np.vstack(X_obs_list)
        dt = np.concatenate(dt_list)
        cog = np.vstack(cog_list)
        ids = np.concatenate(ids_list)

        n_patients = len(unique_ids)

        if initial_beta_list:
            initial_beta = np.array(initial_beta_list)
        else:
            initial_beta = initialize_beta(ids=np.arange(n_patients), beta_range=(0, self.t_max), rng=self.rng)

        K = self.K
        #initial_beta = X.get("initial_beta", None)

        
        rng = self.rng
        self.t_span = np.linspace(0, self.t_max, int(self.t_max / self.step)) # I dont know why this is a class variable might be for plotting
        n_biomarkers = np.size(X_obs,1) # rows = observations, columns = biomarkers
        n_patients = len(np.unique(ids))
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
        initial_f = rng.uniform(0, 0.1, size=n_biomarkers)
        initial_s = rng.uniform(0.1, 3, size=n_biomarkers)
        # initial_scalar_K = float(rng.uniform(0.01, 3, size=1))
        initial_scalar_K = np.max(X_obs)
        initial_theta = np.concatenate([initial_f, initial_s, [initial_scalar_K]])
        
        # cog regression params
        initial_cog_a = np.ones(n_cog_features) # initialize a weight for each type of cog test
        initial_cog_b = 0 # bias term
        
        #initial_cog_a = rng.uniform(1, 5, n_cog_features) # initialize a weight for each type of cog test
        #initial_cog_b = float(rng.uniform(0, 10)) # bias term
        
        ## move these later towards the end for a summary:
        # print("initial conditions:")
        # print(f"n_patients: {n_patients}, n_obs: {n_obs}")
        # print(f"initial f: {initial_f}")
        # print(f"initial s: {initial_s}")
        # print(f"initial scalar K: {initial_scalar_K}")
        # print(f"initial beta: {initial_beta.shape}"," K shape: ", K.shape)
        
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
        #print(X_pred.shape)

        initial_lse = 0.0
        
        for idx, pid in enumerate(np.unique(ids)): # each iter will be like (idx, pid)
            mask = (ids == pid)
            # print(np.size(mask), X_obs.shape, dt.shape, cog.shape)
            X_obs_i, dt_i, cog_i = X_obs[mask,:], dt[mask], cog[mask,:]
            beta_i = initial_beta[idx]
            
            if current_jac:
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
        
        #print(f"prepend complete")
        ### MAIN LOOP ###
        loop_iter = 0
        
        pbar = tqdm(total=self.max_iter)
        while loop_iter < self.max_iter:
            hist_idx = loop_iter + 1
            
            start_theta = time.time()
            current_theta = fit_theta(X_obs=X_obs, dt_obs=dt, ids=ids, K=K,
                                             t_span=self.t_span, use_jacobian=current_jac,
                                             lambda_f=self.lambda_f, lambda_scalar=self.lambda_scalar,
                                             beta_pred=current_beta, f_guess=current_f,
                                             s_guess=current_s, scalar_K_guess=current_scalar_K, rng=rng
                                      )
            
            current_x0, current_f, current_s, current_scalar_K = current_theta
            current_theta = np.concatenate((current_f, current_s, [current_scalar_K]))
            X_pred = solve_system(current_x0, current_f, K, self.t_span, current_scalar_K)
            theta_history[:,hist_idx] = np.concatenate((current_f, current_s, [current_scalar_K]))
            lse = 0.0
            end_theta = time.time()
            
            #print(f"Execution time: {end_theta - start_theta:.4f} seconds")
            
            # TODO: Attempt parallel beta computation
            ### beta comuputaiton
            for idx, patient_id in enumerate(np.unique(ids)):
                mask = (ids == patient_id)
                X_obs_i = X_obs[mask,:]  # (n_obs_i, n_biomarkers)
                dt_i = dt[mask]    # (n_obs_i,)
                cog_i = cog[mask,:]  # (n_obs_i, n_cog_features)
                beta_i = current_beta[idx]
            
                #print("breakpoint 6: ", x_obs_i.shape, x_obs_i.dtype,dt_i.shape, dt_i.dtype, cog_i.shape, cog_i.dtype, beta_i)  
                #print("breakpoint 8: ", current_cog_a.shape)
                #if X_obs_i.shape[0] != dt_i.shape:
                #    print(patient_id, X_obs_i.shape, dt_i.shape)
                    
                beta_i = estimate_beta_for_patient(beta_i=beta_i, X_obs_i=X_obs_i, dt_i=dt_i,
                                                   X_pred=X_pred, t_span=self.t_span,
                                                   cog_i = cog_i, cog_a = current_cog_a, cog_b = current_cog_b,
                                                   theta = current_theta, K=K, lambda_cog = self.lambda_cog,
                                                   use_jacobian = True, t_max = self.t_max
                                                   )
       
                if current_jac == True:
                    res, _ = beta_loss_jac(beta_i, X_obs_i, dt_i, X_pred, self.t_span, cog_i, current_cog_a, current_cog_b, current_theta, K, self.lambda_cog)
                else:
                    res = beta_loss(beta_i, X_obs_i, dt_i, X_pred, self.t_span, cog_i, current_cog_a, current_cog_b, current_theta, self.lambda_cog)
                lse += res
                
                current_beta[idx] = beta_i 
            beta_history[:,hist_idx] = current_beta

            #if self.use_jacobian and lse > best_lse and not jacobian_switched:
            # if self.use_jacobian and best_lse - lse > 1e-3 * best_lse and not jacobian_switched:
            
            epsilon = 1e-2
            relative_tolerance = 1e-3
            delta = best_lse - lse
            
            if (self.jac_toggle == True) and (delta < epsilon):# or lse > best_lse * (1 + relative_tolerance)):
                if jacobian_switched == True: # early convergence detected
                    #print("L-BFGS and Nelder-Mead both failed to improve LSE, exiting early due to convergence")
                    lse_history[hist_idx] = lse
                    break 
                
                current_jac = False
                #print(f"warning: toggling to jac {current_jac}; due to increase or convergence in LSE at iteration {loop_iter}.")
                jacobian_switched = True
                continue
            # if best_lse - lse > 1e-3 * best_lse
                # continue # skip storing values for current iteration. if True --> DONT UPDATE and RETRY
                
            ## update accepted
            jacobian_switched = False
            if self.jac_toggle == True:
                current_jac = True
            
            best_lse = min(best_lse, lse)
            # TODO: Get idk of best lse
            lse_history[hist_idx] = lse
            
            current_cog_a, current_cog_b = fit_linear_cog_regression_multi(cog, dt, current_beta, ids)
            cog_regression_history[:, hist_idx] = np.concatenate([current_cog_a, [current_cog_b]])
            
            loop_iter += 1
            pbar.update(1)
            
            
        self.theta_history = theta_history[:, 0:hist_idx+1]
        self.beta_history = beta_history[:, 0:hist_idx+1]
        self.lse_history = lse_history[0:hist_idx+1]
        self.cog_regression_history = cog_regression_history[:, 0:hist_idx+1]
        #print("fit complete")
        
        return self
    
    def cross_val_lse(self, X):
        cog_a = self.cog_regression_history[:-1, -1]  # shape (n_cog_features,)
        cog_b = self.cog_regression_history[-1, -1]
        theta = self.theta_history[:, -1]

        n_biomarkers = (len(theta) - 1) // 2

        f = theta[0:n_biomarkers]
        s = theta[n_biomarkers:2 * n_biomarkers]
        scalar_K = theta[-1]
        x0 = np.zeros_like(f)

        # Use the model's stored K
        X_model = solve_system(x0, f, self.K, self.t_span, scalar_K)

        total_lse = 0.0

        for patient in X:
            dt_i = patient["dt"]
            cog_i = patient["cog"][0]  # use first visit to estimate beta
            X_obs_i = patient["X_obs"]

            beta_i_guess = cog_a @ cog_i + cog_b
            beta_hat = estimate_beta_for_patient(
                beta_i=beta_i_guess,
                X_obs_i=X_obs_i,
                dt_i=dt_i,
                X_pred=X_model,
                t_span=self.t_span,
                cog_i=cog_i,
                cog_a=cog_a,
                cog_b=cog_b,
                theta=theta,
                K=self.K,
                lambda_cog=self.lambda_cog,
                use_jacobian=self.jac_toggle,
                t_max=self.t_span[-1]
            )
            t_ij = dt_i + beta_hat

            x_pred = np.array([
                np.interp(t_ij, self.t_span, X_model[b_idx])
                for b_idx in range(X_model.shape[0])
            ]).T  # shape (n_visits, n_biomarkers)

            total_lse += np.sum((s[None,:]*x_pred - X_obs_i) ** 2)

        return total_lse



    def score(self, X, y=None):
#        print("[SCORE] score was called")
        try:
            loss = self.cross_val_lse(X)
            #print(f"[SCORE] LSE: {loss:.4f}")
            return -loss
        except Exception as e:
            #print(f"[SCORE ERROR] {e}")
            return float("-inf")
    
    from datetime import datetime

    def save(self, save_path: str = None):
        if save_path == None:
            loc = "/home/dsemchin/Progression_models_simulations/EMDPM/experiments/real_data_PPMI/"
            name = f"ppmi_maxiter{self.max_iter}_jac{self.jac_toggle}_{datetime.now().strftime('%H-%M-%S')}.npz"
            save_path = loc + name
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(save_path,
                            theta_history=self.theta_history,
                            beta_history=self.beta_history,
                            lse_history=self.lse_history,
                            cog_regression_history=self.cog_regression_history)
        print(f"Saved histories to {save_path}.npz")
