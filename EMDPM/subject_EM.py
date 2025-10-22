# subject_em.py
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from EMDPM.optimizer_theta_subject import fit_theta_subject
from EMDPM.utils import solve_system

class SubjectEM(BaseEstimator, TransformerMixin):
    """
    Per-patient theta refitter with fixed betas.
    Expects per-patient dict with keys:
      'X_obs' (n_obs_i, n_biomarkers), 'dt' (n_obs_i,),
      'beta_pred' (float),
      'f_init' (n_biomarkers,), 's_init' (n_biomarkers,), 'scalar_K_init' (float)
    """

    def __init__(self, K=None, t_max=40.0, step=0.01, max_iter=200,
                 use_jacobian=True, lambda_f=1.0, lambda_scalar=0.3, verbose=1):
        self.K = K
        self.t_max = t_max
        self.step = step
        self.max_iter = max_iter
        self.use_jacobian = use_jacobian
        self.lambda_f = lambda_f
        self.lambda_scalar = lambda_scalar
        self.verbose = verbose

        # set during fit
        self.t_span = None
        self.initial_f = None
        self.initial_s = None
        self.initial_scalar_K = None

        self.forcing_matrix_ = None     # (n_patients, n_biomarkers)
        self.s_matrix_ = None           # (n_patients, n_biomarkers)
        self.scalar_K_vec_ = None       # (n_patients,)
        self.delta_forcing_matrix_ = None
        self.delta_s_matrix_ = None
        self.delta_scalar_K_vec_ = None

    def _make_t_span(self):
        n_steps = int(np.round(self.t_max / self.step)) + 1
        return np.linspace(0.0, self.t_max, n_steps)

    def _as_patient_list(self, X):
        # Accept list/tuple or numpy object array of dicts
        if isinstance(X, (list, tuple)):
            return X
        if isinstance(X, np.ndarray):
            if X.dtype == object:
                return list(X)  # preserves order
        raise ValueError("X must be a list/tuple or numpy object array of per-patient dicts.")

    def fit(self, X, y=None):
        X = self._as_patient_list(X)
        if len(X) == 0:
            raise ValueError("X is empty.")

        n_patients = len(X)
        n_biomarkers = X[0]["X_obs"].shape[1]
        self.t_span = self._make_t_span()

        # use first patient's inits as reference for deltas
        f0 = np.asarray(X[0]["f_init"], dtype=float).reshape(-1)
        s0 = np.asarray(X[0]["s_init"], dtype=float).reshape(-1)
        scalar_K0 = float(X[0]["scalar_K_init"])
        if f0.shape[0] != n_biomarkers or s0.shape[0] != n_biomarkers:
            raise ValueError(f"Init shapes mismatch: f={f0.shape}, s={s0.shape}, expected ({n_biomarkers},)")

        self.initial_f = f0.copy()
        self.initial_s = s0.copy()
        self.initial_scalar_K = scalar_K0

        # allocate outputs
        self.forcing_matrix_ = np.zeros((n_patients, n_biomarkers), float)
        self.s_matrix_ = np.zeros((n_patients, n_biomarkers), float)
        self.scalar_K_vec_ = np.zeros(n_patients, float)
        self.delta_forcing_matrix_ = np.zeros((n_patients, n_biomarkers), float)
        self.delta_s_matrix_ = np.zeros((n_patients, n_biomarkers), float)
        self.delta_scalar_K_vec_ = np.zeros(n_patients, float)

        iterator = tqdm(range(n_patients), desc="Fitting subject thetas") if self.verbose >= 1 else range(n_patients)

        for i in iterator:
            p = X[i]
            X_obs_i = np.asarray(p["X_obs"], float)
            dt_i = np.asarray(p["dt"], float)
            beta_i = float(p["beta_pred"])

            init_f = np.asarray(p["f_init"], float).reshape(-1)
            init_s = np.asarray(p["s_init"], float).reshape(-1)
            init_scalar_K = float(p["scalar_K_init"])

            cur_f, cur_s, cur_scalar_K = fit_theta_subject(
                X_obs_i=X_obs_i,
                dt_i=dt_i,
                beta_i=beta_i,
                K=self.K,
                t_span=self.t_span,
                use_jacobian=self.use_jacobian,
                lambda_f=self.lambda_f,
                lambda_scalar=self.lambda_scalar,
                f_init=init_f,
                s_init=init_s,
                scalar_K_init=init_scalar_K,
                #max_iter=self.max_iter,
            )

            # store per-patient finals
            self.forcing_matrix_[i, :] = cur_f
            self.s_matrix_[i, :] = cur_s
            self.scalar_K_vec_[i] = cur_scalar_K

            # store deltas vs inits
            self.delta_forcing_matrix_[i, :] = cur_f - init_f
            self.delta_s_matrix_[i, :] = cur_s - init_s
            self.delta_scalar_K_vec_[i] = cur_scalar_K - init_scalar_K

            # write back to patient dict in-place
            p["final_f"] = cur_f.copy()
            p["final_s"] = cur_s.copy()
            p["final_scalar_K"] = float(cur_scalar_K)
            p["delta_f"] = (cur_f - init_f).copy()
            p["delta_s"] = (cur_s - init_s).copy()
            p["delta_scalar_K"] = float(cur_scalar_K - init_scalar_K)

            # always compute per-patient trajectory
            p["X_pred_subject"] = solve_system(np.zeros(n_biomarkers), cur_f, self.K, self.t_span, cur_scalar_K)

        if self.verbose >= 1:
            mean_abs_df = float(np.mean(np.abs(self.delta_forcing_matrix_)))
            print(f"SubjectEM completed. mean |Δf| = {mean_abs_df:.6f}")

        return self

    def transform(self, X=None):
        if self.forcing_matrix_ is None:
            raise RuntimeError("fit() must be run before transform().")
        return {
            "f": self.forcing_matrix_,
            "s": self.s_matrix_,
            "scalar_K": self.scalar_K_vec_,
            "delta_f": self.delta_forcing_matrix_,
            "delta_s": self.delta_s_matrix_,
            "delta_scalar_K": self.delta_scalar_K_vec_,
        }
