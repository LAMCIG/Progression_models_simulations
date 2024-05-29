from biomarker_utils import sigmoid_inv, generate_transition_matrix, apply_transition_matrix2, solve_ode_system, multi_logistic_deriv, random_connectivity_matrix
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class BiomarkerSimulation:
    def __init__(self, n_patients_stage: dict, biomarkers_params: dict, method: str = 'sigmoid_inv', add_noise: bool = False, noise_std: float = 0.1, random_state: int = None, n_stages: int = 40):
        """
        Initializes the BiomarkerSimulation class.

        Parameters:
        n_patients_stage (dict): Dictionary specifying the number of patients at each disease stage.
        biomarkers_params (dict): Dictionary specifying the parameters for each biomarker.
        method (str): The simulation method to use ('sigmoid_inv', 'transition_matrix', 'ode').
        add_noise (bool): Whether to add Gaussian noise to the biomarker values.
        noise_std (float): The standard deviation of the Gaussian noise.
        random_state (int): Seed for the random number generator for reproducibility.
        n_stages (int): The number of stages to simulate.
        """
        self.n_patients_stage = n_patients_stage
        self.biomarkers_params = biomarkers_params
        self.method = method
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.random_state = random_state
        self.n_stages = n_stages  # Number of stages for plotting
        self.X = None
        self.y = None
        self.t = None  # iime points for ODE
        self.methods = {
            'sigmoid_inv': self.generate_patient_sigmoid,
            'transition_matrix': self.generate_patient_transition,
            'ode': self.generate_patient_ode
        }

    def generate_patient(self, stage):
        """Delegates to the specific method based on the simulation type."""
        return self.methods[self.method](stage)

    def generate_patient_sigmoid(self, stage):
        """
        Generate patient biomarkers using an inverse sigmoid method.

        Parameters:
        stage (int): The disease stage of the patient.

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

    def simulate(self):
        if self.method == 'ode':
            self.t, self.X = self.generate_patient_ode()
            self.stages = np.zeros(self.X.shape[0])  # Placeholder for stages,
            threshold = 0.8  # Threshold of disease
            disease_biomarker_index = 0  # Index of the biomarker used to determine health status
            self.y = (self.X[:, disease_biomarker_index] > threshold).astype(int)
        else:
            X = []
            stages = []
            for stage in range(self.n_stages):
                for _ in range(self.n_patients_stage.get(stage, 0)):
                    patient_biomarkers = self.generate_patient(stage)
                    X.append(patient_biomarkers)
                    stages.append(stage)
            self.X = np.array(X)
            self.stages = np.array(stages)
            n_healthy = sum(1 for stage in self.stages if stage <= 3)
            n_diseased = len(self.stages) - n_healthy
            self.y = np.array([0] * n_healthy + [1] * n_diseased)
        return self.X, self.y, self.stages

    def plot_biomarkers(self):
        """Plots the progression of biomarkers over stages."""
        if self.X is None or self.stages is None:
            raise ValueError("Simulation data not available. Run simulate() first.")
        
        plt.figure(figsize=(12, 8))
        for i in range(self.X.shape[1]):
            plt.plot(self.stages, self.X[:, i], label=f'Biomarker {i+1}')
        plt.xlabel('Stage')
        plt.ylabel('Biomarker Value')
        plt.legend()
        plt.show()
    
    def plot_patient_stages(self):
        """Plots the progression of biomarkers with patient stages superimposed."""
        if self.X is None or self.stages is None:
            raise ValueError("Simulation data not available. Run simulate() first.")
        
        plt.figure(figsize=(12, 8))
        for i in range(self.X.shape[1]):
            plt.plot(self.stages, self.X[:, i], label=f'Biomarker {i+1}', alpha=0.3)
        plt.scatter(self.stages, self.X[:, 0], label='Patient Stages', color='red', s=10)
        plt.xlabel('Stage')
        plt.ylabel('Biomarker Value')
        plt.legend()
        plt.show()
    
    def plot_biomarker_distribution(self, biomarker_index):
        """Plots the distribution of a specific biomarker for healthy and unhealthy patients."""
        if self.X is None or self.y is None:
            raise ValueError("Simulation data not available. Run simulate() first.")
        
        biomarker_values = self.X[:, biomarker_index]
        n_healthy = sum(self.y == 0)
        n_diseased = sum(self.y == 1)
        
        bins = plt.hist(biomarker_values, alpha=0.1, label='all', bins=20)
        plt.hist(biomarker_values[:n_healthy], label='healthy', bins=bins[1], alpha=0.3)
        plt.hist(biomarker_values[n_healthy:], label='diseased', bins=bins[1], alpha=0.3)
        plt.legend()
        plt.xlabel('Biomarker Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Biomarker {biomarker_index + 1}')
        plt.show()
