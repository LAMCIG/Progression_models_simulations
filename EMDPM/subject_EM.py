# subject_em.py
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from EMDPM.optimizer_theta_subject import fit_theta_subject

class SubjectEM(BaseEstimator, TransformerMixin):
    """
    Per-patient theta refitter with fixed betas for subtyping.
    Expects X to be a list of dicts with keys: 'X_obs', 'dt', 'beta', 'theta_init'.
    """

    def __init__(self,
                 K=None,
                 max_iter=200,
                 t_max=None,
                 save_delta=True,
                 verbose=1):
        self.K = K
        self.max_iter = max_iter
        self.t_max = t_max
        self.save_delta = save_delta
        self.verbose = verbose

        # will be set in fit
        self.initial_f = None
        self.initial_s = None
        self.initial_scalar_K = None

        self.current_f = None
        self.current_s = None
        self.current_scalar_K = None

        self.forcing_matrix_ = None          # shape (n_patients, n_biomarkers)
        self.delta_forcing_matrix_ = None    # shape (n_patients, n_biomarkers) if save_delta

    def fit(self, X, y=None):
        if not isinstance(X, (list, tuple)) or len(X) == 0:
            raise ValueError("X must be a non-empty list of per-patient dicts.")

        n_patients = len(X)
        n_biomarkers = X[0]["X_obs"].shape[1]

        # get theta from the first patient and split into f, s, scalar_K
        theta0 = np.array(X[0]["theta_init"], dtype=float)
        f0 = theta0[0:n_biomarkers]
        s0 = theta0[n_biomarkers:2*n_biomarkers]
        scalar_K0 = theta0[-1]

        self.initial_f = f0
        self.initial_s = s0
        self.initial_scalar_K = scalar_K0

        self.current_f = f0.copy()
        self.current_s = s0.copy()
        self.current_scalar_K = scalar_K0

        # preallocate outputs
        self.forcing_matrix_ = np.zeros((n_patients, n_biomarkers), dtype=float)
        if self.save_delta:
            self.delta_forcing_matrix_ = np.zeros((n_patients, n_biomarkers), dtype=float)
        else:
            self.delta_forcing_matrix_ = None

        if self.verbose >= 1:
            iterator = tqdm(range(n_patients), desc="Fitting subject thetas")
        else:
            iterator = range(n_patients)

        for i in iterator:
            p = X[i]
            X_obs_i = np.asarray(p["X_obs"], dtype=float)
            dt_i = np.asarray(p["dt"], dtype=float)
            beta_i = float(p["beta"])

            # same initialization for every patient
            current_f = f0.copy()
            current_s = s0.copy()
            current_scalar_K = float(scalar_K0)

            current_f, current_s, current_scalar_K = fit_theta_subject(
                X_obs_i=X_obs_i,
                dt_i=dt_i,
                beta_i=beta_i,
                K=self.K,
                t_span=self.t_span,
                use_jacobian=True,
                lambda_f=1.0,
                lambda_scalar=0.3,
                f_init=self.initial_f,
                s_init=self.initial_s,
                scalar_K_init=self.initial_scalar_K,
            )
            # store results
            self.forcing_matrix_[i, :] = current_f
            if self.delta_forcing_matrix_ is not None:
                self.delta_forcing_matrix_[i, :] = current_f - f0 # TODO: do a different kind of transfor,

        if self.verbose >= 1:
            print("SubjectEM completed.")
            if self.delta_forcing_matrix_ is not None:
                print(f"mean abs delta f: {np.mean(np.abs(self.delta_forcing_matrix_)):.6f}")

        return self

    def transform(self, X=None):
        if self.forcing_matrix_ is None:
            raise RuntimeError("fit() must be run before transform().")
        return self.delta_forcing_matrix_ if self.delta_forcing_matrix_ is not None else self.forcing_matrix_
