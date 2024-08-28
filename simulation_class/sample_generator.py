import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.stats import norm, uniform, erlang, pareto

class SampleGenerator:
    """
    Sample generator takes in stage values and parameters for generating a patient sample.
    
    Parameters:
        stage_values (np.ndarray): 2D array of biomarker values, where rows are biomarkers and columns are stages.
        n_patients (int): Number of patients in your sample.
        distribution (scipy.stats distribution): Distribution to sample patient stages.
        dist_params (dict): Parameters for the chosen distribution.
        add_noise (bool): Adds noise if True.
        noise_std (float): The standard deviation of the noise.
        random_state (int): Set random state for reproducible pseudo-random results.
    
    Attributes:
        patient_samples (list): List of tuples with stage and biomarker values for each patient.
    """
    def __init__(self, stage_values: np.ndarray, n_patients: int, distribution=norm, dist_params=None, add_noise: bool = True, noise_std: float = 0.1, random_state: int = None):
        self.stage_values = stage_values
        self.n_patients = n_patients
        self.distribution = distribution
        self.dist_params = dist_params if dist_params is not None else {}
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.random_state = random_state
        self.patient_samples = self._generate_patient_samples()

    def _generate_patient_samples(self) -> List[Tuple[int, np.ndarray]]:
        random = np.random.RandomState(self.random_state)
        patient_samples = []
        
        stages = self._generate_stages()
        for stage in stages:
            biomarkers = self.stage_values[:, stage]
            if self.add_noise:
                noise = random.normal(0, self.noise_std, biomarkers.shape)
                biomarkers = np.clip(biomarkers + noise, 0, 1)  # ensures values are in the range [0,1]
            patient_samples.append((stage, biomarkers))
        return patient_samples
    
    def _generate_stages(self) -> np.ndarray:
        random = np.random.RandomState(self.random_state)
        n_stages = self.stage_values.shape[1]
        
        # Generate stages based on the specified distribution
        stages = self.distribution.rvs(size=self.n_patients, **self.dist_params)
        
        # Clip and round stages to ensure they fall within the correct range
        stages = np.clip(np.round(stages), 0, n_stages - 1).astype(int)
        return stages

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        stages = np.array([sample[0] for sample in self.patient_samples])
        biomarkers = np.array([sample[1] for sample in self.patient_samples])
        n_healthy = sum(stages <= 3)  # adjust threshold as needed
        y = np.array([0] * n_healthy + [1] * (len(stages) - n_healthy))  # recall this only works when patients are ordered by biomarker
        return biomarkers, y

    # Getter method for X and y
    def get_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._prepare_data()

    #%% Plotting methods
    def plot_stage_histogram(self) -> None:
        stages = [sample[0] for sample in self.patient_samples]
        plt.hist(stages, bins=self.stage_values.shape[1], alpha=0.5)
        plt.xlabel('Disease Stage')
        plt.ylabel('Number of Patients')
        plt.title('Patient Stage Distribution')
        plt.show()

    def plot_biomarker_distribution(self, biomarker_index: int, healthy_stage_threshold: float = 2) -> None:
        biomarkers = np.array([sample[1][biomarker_index] for sample in self.patient_samples])
        stages = np.array([sample[0] for sample in self.patient_samples])
        
        healthy = biomarkers[stages <= healthy_stage_threshold * self.stage_values.shape[1]]
        unhealthy = biomarkers[stages > healthy_stage_threshold * self.stage_values.shape[1]]
        
        bins = plt.hist(biomarkers, alpha=0.1, label='All', bins=20)
        plt.hist(healthy, bins=bins[1], alpha=0.5, label='Healthy')
        plt.hist(unhealthy, bins=bins[1], alpha=0.5, label='Unhealthy')
        plt.xlabel('Biomarker Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Biomarker {biomarker_index + 1}')
        plt.legend()
        plt.show()

    def plot_patient_biomarkers(self, patient_index: int) -> None:
        patient_sample = self.patient_samples[patient_index]
        stage, biomarkers = patient_sample
        plt.scatter([stage] * len(biomarkers), biomarkers)
        plt.xlabel('Disease Stage')
        plt.ylabel('Biomarker Value')
        plt.title(f'Biomarker Values for Patient {patient_index}')
        plt.show()
