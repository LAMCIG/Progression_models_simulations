from biomarker_utils import sigmoid_inv, generate_transition_matrix, apply_transition_matrix2, solve_ode_system, multi_logistic_deriv, random_connectivity_matrix
import numpy as np

class BiomarkerSimulation:
    def __init__(self, n_patients_stage, biomarkers_params, method='sigmoid_inv', **kwargs):
        self.n_patients_stage = n_patients_stage
        self.biomarkers_params = biomarkers_params
        self.method = method
        self.params = kwargs
        self.X = None
        self.y = None
        self.methods = {
            'sigmoid_inv': self.generate_patient_sigmoid,
            'transition_matrix': self.generate_patient_transition,
            'ode': self.generate_patient_ode
        }

    def generate_patient(self, stage, add_noise=False, noise_std=0.1, random_state=None):
        if self.method == 'sigmoid_inv':
            return self.generate_patient_sigmoid(stage, add_noise=add_noise, noise_std=noise_std, random_state=random_state)
        elif self.method == 'transition_matrix':
            return self.generate_patient_transition(stage, add_noise=add_noise, noise_std=noise_std, random_state=random_state)

    def generate_patient_sigmoid(self, stage, add_noise=False, noise_std=0.1, random_state=None):
        """
        Generate patient biomarkers using an inverse sigmoid method.

        Parameters:
        stage (int): The disease stage of the patient.
        add_noise (bool): If True, adds Gaussian noise to the biomarkers.
        noise_std (float): The standard deviation of the Gaussian noise.
        random_state (int): Seed for the random number generator for reproducibility.

        Returns:
        np.ndarray: An array of biomarker values for a single patient.
        """
        random = np.random.RandomState(random_state)
        x = []
        for marker, parameters in self.biomarkers_params.items():
            value = sigmoid_inv(stage, parameters['s'], parameters['c'])
            if add_noise:
                value += random.normal(0, noise_std)
            value = np.clip(value, 0, 1)
            x.append(value)
        return np.array(x)

    def generate_patient_transition(self, stage, add_noise=False, noise_std=0.1, random_state=None):
        """Generate patient biomarkers using transition matrix method."""
        transition_matrix = self.biomarkers_params['transition_matrix']
        y_init = self.biomarkers_params['y_init']
        y_stage = apply_transition_matrix2(transition_matrix, stage, y_init)
        if add_noise:
            random = np.random.RandomState(random_state)
            noise = random.normal(0, noise_std, y_stage.size)
            y_stage += noise
        y_stage = np.clip(y_stage, 0, 1)
        return y_stage
    
    def generate_patient_ode(self):
        # from dict
        K = self.biomarkers_params['connectivity_matrix']
        y_init = self.biomarkers_params['y_init']
        t_span = self.biomarkers_params['t_span']
        n_steps = self.biomarkers_params['n_steps']
        x0 = np.zeros(len(y_init))
        x0[0] = 1  # init condition
        t, x = solve_ode_system(K, x0, t_span, n_steps)
        return t, x

    # def simulate(self):
    #     if self.method in self.methods:
    #         return self.methods[self.method]()
    #     else:
    #         raise ValueError(f"Method '{self.method}' not implemented.")

    def simulate(self):
        if self.method == 'ode':
            return self.generate_patient_ode()
        else:
            X = []
            stages = []
            rs = self.params.get('random_state', 0)
            for stage, total_number in self.n_patients_stage.items():
                for _ in range(total_number):
                    patient_biomarkers = self.generate_patient(stage, add_noise=True, noise_std=0.5, random_state=rs)
                    X.append(patient_biomarkers)
                    stages.append(stage)
            X = np.array(X)
            n_healthy = sum(v for k, v in self.n_patients_stage.items() if k <= 3)
            n_diseased = sum(self.n_patients_stage.values()) - n_healthy
            y = np.array([0] * n_healthy + [1] * n_diseased)
            self.X = X
            self.y = y
            return X, y, np.array(stages)
