import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from .synthetic_data_generator import initialize_beta
from .optimizer_theta import fit_theta
from .optimizer_beta import estimate_beta_for_patient, beta_loss, beta_loss_jac
from .utils import solve_system

class EM(BaseEstimator, TransformerMixin):
    """
    EM algorithm for recovering disease progression model parameters
    and patient-specific time shifts from cross-sectional observations.
    """

    def __init__(self, 
                 num_iterations: int = 50,
                 t_max: float = 12, step: float = 0.01,
                 K: np.ndarray = None, use_jacobian: bool = False,
                 lamda: float = 0.01,
                 alpha: float = 1 # scalar param on matrix
                 ):
        # separate attributes and paramaters
        
        self.num_iterations = num_iterations
        self.t_max = t_max
        self.step = step
        self.K = K
        self.use_jacobian = use_jacobian
        self.lamda = lamda # TODO: consider refactoring to like "lasso_penalty" or something, ask BG
        self.alpha = alpha
        
        ## attributes for switching logic
        self.best_lse = np.inf
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Run the EM loop to estimate theta (ODE params) and beta (patient shifts).

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe containing patient observations.
        y : Ignored

        Returns
        -------
        self
        """
        if self.K is None:
            raise ValueError("Connectivity matrix K must be provided at initialization.")

        self.t_span = np.linspace(0, self.t_max, int(self.t_max / self.step))
        n_biomarkers = len([col for col in X.columns if "biomarker_" in col])

        self.beta_iter_ = initialize_beta(X, beta_range=(0, self.t_max))
        self.theta_iter_ = pd.DataFrame(columns=[f"iter_{i}" for i in range(self.num_iterations)])
        self.lse_array_ = []

        df_copy = X.copy()
        self.K_ = self.K
        best_lse = np.inf # keep outside loop or else it resets
        jacobian_switched = False


        iteration = 0
        pbar = tqdm(total=self.num_iterations)
        while iteration < self.num_iterations:
            x0_fit, f_fit = fit_theta(df_copy, 
                                      self.beta_iter_,
                                      iteration,
                                      self.K_,
                                      self.t_span,
                                      use_jacobian=self.use_jacobian,
                                      lamda = self.lamda
                                      )
            x_reconstructed = solve_system(x0_fit, f_fit, self.K_, self.t_span)

            beta_estimates = {}
            lse = 0
            # best_lse = np.inf
            
            # TODO: Attempt parallel beta computation
            ### beta comuputaiton
            for patient_id, df_patient in df_copy.groupby("patient_id"):
                beta_i = estimate_beta_for_patient(df_patient, x_reconstructed, self.t_span, self.t_max, use_jacobian=self.use_jacobian)
                beta_estimates[patient_id] = beta_i

                x_obs = df_patient[[col for col in df_patient.columns if "biomarker_" in col]].values.T
                
                if self.use_jacobian == True:
                    res, _ = beta_loss_jac(beta_i, df_patient["dt"].values, x_obs, x_reconstructed, self.t_span)
                else:
                    res = beta_loss(beta_i, df_patient["dt"].values, x_obs, x_reconstructed, self.t_span)
                lse += res

            if self.use_jacobian and lse > best_lse and not jacobian_switched:
                print(f"warning: jacobian toggled off due to increase in LSE at iteration {iteration}.")
                self.use_jacobian = False
                jacobian_switched = True
                
                # this prevents loop from crashing due to missing column
                current_col = str(iteration)
                previous_col = str(iteration - 1)

                if current_col not in self.beta_iter_.columns:
                    self.beta_iter_[current_col] = self.beta_iter_[previous_col]

                df_copy["beta"] = self.beta_iter_[current_col]
                df_copy["t_ij"] = df_copy["beta"] + df_copy["dt"]

                continue # skip storing values for current iteration. if True --> DONT UPDATE and RETRY
            
            ## update accepted
            best_lse = min(best_lse, lse)
            
            # update theta and lse
            self.theta_iter_[f"iter_{iteration}"] = np.concatenate([x0_fit, f_fit])
            self.lse_array_.append(lse)
            
            # update beta
            beta_update_df = pd.DataFrame(list(beta_estimates.items()), columns=["patient_id", str(iteration + 1)])
            self.beta_iter_ = self.beta_iter_.merge(beta_update_df, on="patient_id", how="left")

            # recompute t_ij
            df_copy["beta"] = self.beta_iter_[str(iteration)]
            df_copy["t_ij"] = df_copy["beta"] + df_copy["dt"]
                
            iteration += 1
            pbar.update(1)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Append final estimated beta and t_ij to original DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Original input DataFrame.

        Returns
        -------
        pd.DataFrame
            Updated DataFrame with final beta and t_ij columns.
        """
        df_copy = X.copy()
        final_iter = str(self.num_iterations - 1)
        df_copy["beta"] = self.beta_iter_[final_iter]
        df_copy["t_ij"] = df_copy["beta"] + df_copy["dt"]
        return df_copy
