import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from .synthetic_data_generator import initialize_beta
from .optimizer_theta import fit_theta
from .optimizer_cognitive_regression import fit_optimizer_regression
from .optimizer_beta import estimate_beta_for_patient, beta_loss, beta_loss_jac
from .utils import solve_system

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

        self.num_iterations = num_iterations
        self.t_max = t_max
        self.step = step
        self.use_jacobian = use_jacobian
        self.rng = np.random.default_rng(75)
        
        # hyperparameters
        self.lamda = lamda # TODO: consider refactoring to like "lasso_penalty" or "lambda_connectiviy"
        self.lambda_cog = lambda_cog
        
        ## attributes for switching logic
        self.best_lse = np.inf
        

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
        if self.K is None:
            raise ValueError("Connectivity matrix K must be provided at initialization.")
        
        rng = self.rng

        self.t_span = np.linspace(0, self.t_max, int(self.t_max / self.step))
        n_biomarkers = len([col for col in X.columns if "biomarker_" in col])

        self.beta_iter_ = initialize_beta(X, beta_range=(0, self.t_max), rng=rng)
        self.theta_iter_ = pd.DataFrame(columns=[f"iter_{i}" for i in range(self.num_iterations + 1)])
        self.lse_array_ = []
        self.cog_regression_history_ = pd.DataFrame(columns=["a", "b"])

        df_copy = X.copy()
        self.K_ = self.K
        best_lse = np.inf # keep outside loop or else it resets
        jacobian_switched = False
        
        ## prepend computation && param intialization
        initial_f = rng.uniform(0, 0.2, size=n_biomarkers)
        initial_s = rng.uniform(0.1, 3, size=n_biomarkers)
        initial_scalar_K = float(rng.uniform(0.01, 3, size=1))
        
        cog_a = float(rng.uniform(1, 5))
        cog_b = float(rng.uniform(0, 10))
        
        print("initial conditions:")
        print(f"initial f: {initial_f}")
        print(f"initial s: {initial_s}")
        print(f"initial scalar: {initial_scalar_K}")
        print(f"initial beta estimates (first 5 patients): {self.beta_iter_[["patient_id", "0"]].head(5)}")
        print(f"initial beta mean: {self.beta_iter_["0"].mean():.4f}")
        x0_initial = np.zeros(n_biomarkers)
        x_reconstructed_init = solve_system(x0_initial, initial_f, self.K_, self.t_span, scalar_K=initial_scalar_K)
        
        # Compute initial LSE using the initialized beta_iter_ (without re-estimating beta)
        df_copy["beta"] = self.beta_iter_["0"]
        df_copy["t_ij"] = df_copy["beta"] + df_copy["dt"]

        initial_lse = 0
        for patient_id, df_patient in df_copy.groupby("patient_id"):
            beta_i = df_patient["beta"].iloc[0]
            x_obs = df_patient[[col for col in df_patient.columns if "biomarker_" in col]].values.T
            s_ij = df_patient["cognitive_score"].values
            
            if self.use_jacobian:
                res, _ = beta_loss_jac(beta_i, df_patient["dt"].values, x_obs,
                                       x_reconstructed_init, self.t_span,
                                       self.lambda_cog, s_ij, cog_a, cog_b)
            else:
                res = beta_loss(beta_i, df_patient["dt"].values, x_obs,
                                x_reconstructed_init, self.t_span,
                                self.lambda_cog, s_ij, cog_a, cog_b)
            initial_lse += res
            
        initial_theta = np.concatenate([x0_initial, initial_f, initial_s, [initial_scalar_K]])
        self.theta_iter_["iter_0"] = initial_theta
        self.lse_array_.append(initial_lse)
        self.cog_regression_history_.loc[0] = [cog_a, cog_b]
            
        ## main loop

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