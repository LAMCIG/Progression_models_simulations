import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from .optimizer_theta import fit_theta
from .optimizer_beta import estimate_beta_for_patient, beta_loss, beta_loss_jac
from .optimizer_cognitive_regression import fit_optimizer_regression
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
        
        self.lse_array_ = []
        
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
        initial_cog_a = rng.uniform(1, 5, n_cog_features) # initialize a weight for each type of cog test
        initial_cog_b = float(rng.uniform(0, 10)) # bias term
        
        ## move these later towards the end for a summary:
        print("initial conditions:")
        print(f"n_patients: {n_patients}, n_obs: {n_obs}")
        print(f"initial f: {initial_f}")
        print(f"initial s: {initial_s}")
        print(f"initial scalar K: {initial_scalar_K}")
        print(f"initial beta: {initial_beta.shape}")
        
        ## initialize histories
        theta_history = np.zeros((np.size(initial_theta), self.num_iterations + 1)) # extra column added for initial guesses
        beta_history = np.zeros((n_patients, self.num_iterations + 1))
        lse_history = np.zeros((n_patients, self.num_iterations + 1))
        cog_regression_history = np.zeros((n_cog_features + 1, self.num_iterations + 1))  # + 1 is for intercept
        
        ## Append initial values to histories
        theta_history[:, 0] = initial_theta
        beta_history[:, 0] = initial_beta
        cog_regression_history[:, 0] = np.concatenate([initial_cog_a, [initial_cog_b]])
        
        ## Compute Initial LSE
        X_pred = solve_system(initial_x0, initial_f, K, self.t_span, scalar_K=initial_scalar_K)
        print(X_pred.shape)

        initial_lse = 0.0
        
        for idx, pid in enumerate(np.unique(ids)): # each iter will be like (idx, pid)
            mask = (ids == pid)
            print(np.size(mask), X.shape, dt.shape, cog.shape)
            X_obs_i, dt_i, cog_i = X[mask], dt[mask], cog.T[mask]
            beta_i = initial_beta[idx]
            
            if self.use_jacobian:
                res, _ = beta_loss_jac(beta_i, X_obs_i, dt_i, X_pred, self.t_span, cog_i, initial_cog_a, initial_cog_b, initial_theta, K)
            else:
                res = beta_loss(beta_i, X_obs_i, dt_i, X_pred, self.t_span, cog_i, initial_cog_a, initial_cog_b)
            initial_lse += res
            
        print(f"LSE Prepend complete! LSE: {initial_lse}")
        lse_history[0] = initial_lse
        
        
        ## MAIN LOOP

        iteration = 1
        pbar = tqdm(total=self.num_iterations)
        while iteration < self.num_iterations:
            prev_col = str(iteration - 1)
            
            if prev_col not in self.beta_iter_.columns:
                raise KeyError(f"Missing expected column '{prev_col}' in beta_iter_ at iteration {iteration}.")
            
            df_copy["beta"] = self.beta_iter_[prev_col]
            df_copy["t_ij"] = df_copy["beta"] + df_copy["dt"]

            #print(f"[DEBUG] Available beta columns: {self.beta_iter_.columns.tolist()}")
            current_model_params = fit_theta(df_copy, 
                                      self.beta_iter_,
                                      iteration - 1, # need to use beta from last completed iter therefore "current - 1"
                                      self.K_,
                                      self.t_span,
                                      use_jacobian=self.use_jacobian,
                                      lamda = self.lamda,
                                      scalar_K_guess = initial_scalar_K,
                                      f_guess = initial_f,
                                      s_guess = initial_s,
                                      rng = rng
                                      )
            
            x0_fit, f_fit, s_fit, scalar_K_fit = current_model_params
            
            x_reconstructed = solve_system(x0_fit, f_fit, self.K_, self.t_span, scalar_K_fit)
            beta_estimates = {}
            lse = 0
            # best_lse = np.inf
            
            # TODO: Attempt parallel beta computation
            ### beta comuputaiton
            for patient_id, df_patient in df_copy.groupby("patient_id"):
                beta_i_prev = self.beta_iter_.loc[self.beta_iter_["patient_id"] == patient_id, prev_col].values[0]
                
                beta_i = estimate_beta_for_patient(beta_i= beta_i_prev,
                                                   df_patient = df_patient,
                                                   x_reconstructed = x_reconstructed,
                                                   t_span= self.t_span,
                                                   t_max = self.t_max,
                                                   use_jacobian=self.use_jacobian,
                                                   lambda_cog = self.lambda_cog,
                                                   a=cog_a, b=cog_b)
                beta_estimates[patient_id] = beta_i

                x_obs = df_patient[[col for col in df_patient.columns if "biomarker_" in col]].values.T
                
                if self.use_jacobian == True:
                    res, _ = beta_loss_jac(beta_i, df_patient["dt"].values, x_obs,
                                           x_reconstructed, self.t_span,
                                           self.lambda_cog, s_ij, cog_a, cog_b)
                else:
                    res = beta_loss(beta_i, df_patient["dt"].values, x_obs,
                                    x_reconstructed, self.t_span,
                                    self.lambda_cog, s_ij, cog_a, cog_b)
                lse += res

            if self.use_jacobian and lse > best_lse and not jacobian_switched:
                print(f"warning: jacobian toggled off due to increase in LSE at iteration {iteration}.")
                self.use_jacobian = False
                jacobian_switched = True
                
                # continue # skip storing values for current iteration. if True --> DONT UPDATE and RETRY
                
            ## update accepted
            best_lse = min(best_lse, lse)
            
            # update theta and lse
            current_theta = np.concatenate([x0_fit, f_fit, s_fit, [scalar_K_fit]])
            self.theta_iter_[f"iter_{iteration}"] = current_theta
            self.lse_array_.append(lse)
            
            # update beta
            beta_update_df = pd.DataFrame(list(beta_estimates.items()), columns=["patient_id", str(iteration)])
            self.beta_iter_ = self.beta_iter_.merge(beta_update_df, on="patient_id", how="left")
            
            ## update cog regression parameters
            df_copy["beta"] = self.beta_iter_[str(iteration)]
            df_copy["t_ij"] = df_copy["dt"] + df_copy["beta"]
            
            cog_a, cog_b = fit_optimizer_regression(df_copy,
                                                    self.beta_iter_,
                                                    iteration,
                                                    x_reconstructed,
                                                    self.t_span,
                                                    self.lambda_cog,
                                                    s_fit,
                                                    use_jacobian=self.use_jacobian,
                                                    a_guess=cog_a, b_guess=cog_b
                                                    )
            
            self.cog_regression_history_.loc[iteration] = [cog_a, cog_b]
                
            iteration += 1
            pbar.update(1)
            
            # Summary
            best_iter = np.argmin(self.lse_array_)
            initial_theta = self.theta_iter_["iter_0"].values
            final_theta = self.theta_iter_[f"iter_{self.num_iterations - 1}"].values
            best_theta = self.theta_iter_[f"iter_{best_iter}"].values

        print("\nSUMMARY:")
        print(f"best LSE at iteration {best_iter}: {self.lse_array_[best_iter]}")
        print("initial theta")
        print(initial_theta)
        print("best theta:")
        print(best_theta)
        print("final theta:")
        print(final_theta)

        return self