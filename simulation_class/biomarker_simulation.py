from biomarker_utils import sigmoid_inv, generate_transition_matrix, apply_transition_matrix2, solve_ode_system, multi_logistic_deriv, random_connectivity_matrix
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class BiomarkerSimulation:
    def __init__(self, n_patients_stage, biomarkers_params, method='sigmoid_inv', add_noise=False, noise_std=0.1, random_state=None):
        self.n_patients_stage = n_patients_stage
        self.biomarkers_params = biomarkers_params
        self.method = method
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.random_state = random_state
        self.X = None
        self.y = None
        self.t = None  # iime points for ODE
        self.methods = {
            'sigmoid_inv': self.generate_patient_sigmoid,
            'transition_matrix': self.generate_patient_transition,
            'ode': self.generate_patient_ode
        }

    def generate_patient(self, stage):
        """delegates to the specific method based on the simulation type."""
        return self.methods[self.method](stage)

    def generate_patient_sigmoid(self, stage):
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
        random = np.random.RandomState(self.random_state)
        x = []
        for marker, parameters in self.biomarkers_params.items():
            value = sigmoid_inv(stage, parameters['s'], parameters['c'])
            if self.add_noise:
                value += random.normal(0, self.noise_std)
            value = np.clip(value, 0, 1)
            x.append(value)
        return np.array(x)

    def generate_patient_transition(self, stage):
        """Generate patient biomarkers using transition matrix method."""
        transition_matrix = self.biomarkers_params['transition_matrix']
        y_init = self.biomarkers_params['y_init']
        y_stage = apply_transition_matrix2(transition_matrix, stage, y_init)
        if self.add_noise:
            random = np.random.RandomState(self.random_state)
            noise = random.normal(0, self.noise_std, y_stage.size)
            y_stage += noise
        y_stage = np.clip(y_stage, 0, 1)
        return y_stage
    
    def generate_patient_ode(self):
        """Simulate biomarkers using an ODE model."""
        def ode_system(t, y):
            return -0.1 * y  # exponential decay model

        y_init = self.biomarkers_params['y_init']
        t_span = self.biomarkers_params['t_span']
        t_eval = np.linspace(t_span[0], t_span[1], self.biomarkers_params['n_steps'])

        sol = solve_ivp(ode_system, t_span, y_init, t_eval=t_eval, method='RK45')
        return sol.t, sol.y.T

    # def simulate(self):
    #     if self.method in self.methods:
    #         return self.methods[self.method]()
    #     else:
    #         raise ValueError(f"Method '{self.method}' not implemented.")

    def simulate(self):
        if self.method == 'ode':
            self.t, self.X = self.generate_patient_ode()
            self.stages = np.zeros(self.X.shape[0])  # placeholder for stages,
            threshold = 0.8  # threshold of disease
            disease_biomarker_index = 0  # index of the biomarker used to determine health status
            self.y = (self.X[:, disease_biomarker_index] > threshold).astype(int)
        else:
            X = []
            stages = []
            for stage, total_number in self.n_patients_stage.items():
                for _ in range(total_number):
                    patient_biomarkers = self.generate_patient(stage)
                    X.append(patient_biomarkers)
                    stages.append(stage)
            self.X = np.array(X)
            self.stages = np.array(stages)
            n_healthy = sum(v for k, v in self.n_patients_stage.items() if k <= 3)
            n_diseased = sum(self.n_patients_stage.values()) - n_healthy
            self.y = np.array([0] * n_healthy + [1] * n_diseased)
        return self.X, self.y, self.stages

    def plot_biomarkers(self):
        """Plots the progression of biomarkers over stages."""
        if self.X is None or self.stages is None:
            raise ValueError("Simulation data not available. Run simulate() first.")
        if self.method == 'ode':
            plt.figure(figsize=(10, 5))
            for i in range(self.X.shape[1]):
                plt.plot(self.t, self.X[:, i], label=f'Biomarker {i+1}')
            plt.xlabel('Time')
            plt.ylabel('Biomarker Value')
            plt.title("ODE Biomarker Progression Over Time")
            plt.legend()
            plt.show()
        else:        
            plt.figure(figsize=(12, 8))
            num_biomarkers = self.X.shape[1]
            unique_stages = np.unique(self.stages)
            for i in range(num_biomarkers):
                mean_values = [np.mean(self.X[self.stages == stage, i]) for stage in unique_stages]
                std_values = [np.std(self.X[self.stages == stage, i]) for stage in unique_stages]
                plt.errorbar(unique_stages, mean_values, yerr=std_values, fmt='-o', label=f'Biomarker {i+1}', capsize=5)
            plt.xlabel('Patient Stage')
            plt.ylabel('Biomarker Value')
            plt.title(f'Biomarker Progression Using {self.method} Method')
            plt.legend()
            plt.grid(True)
            plt.show()
        