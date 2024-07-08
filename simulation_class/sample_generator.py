import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class SampleGenerator:
    """
    Sample generator takes in a model (CanonicalGenerator), and paramaters for generating a patient sample.
    
    Parameters:
        canonical_generator (CanonicalGenerator): Pass in a canonical generator after it has been instantiated.
        n_patients (int): Number of patient in your sample.
        add_noise (bool): Adds noise if True.
        noise_std (float):
        random_state (int):
        skewness (float):
        
    Attributes:
        patient_samples ():
    """
    def __init__(self, canonical_generator, n_patients: int, add_noise: bool = True, noise_std: float = 0.1, random_state: int = None, skewness: float = -1):
        self.canonical_generator = canonical_generator
        self.n_patients = n_patients
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.random_state = random_state
        self.skewness = skewness
        self.patient_samples = self._generate_patient_samples()

    def _generate_patient_samples(self) -> List[Tuple[int, np.ndarray]]:
        random = np.random.RandomState(self.random_state)
        patient_samples = []
        
        stages = self._generate_stages()
        for stage in stages:
            biomarkers = self.canonical_generator.biomarker_values[:, stage]
            if self.add_noise:
                noise = random.normal(0, self.noise_std, biomarkers.shape)
                biomarkers = np.clip(biomarkers + noise, 0, 1)
            patient_samples.append((stage, biomarkers))
        return patient_samples
    
    def _generate_stages(self) -> np.ndarray:
        random = np.random.RandomState(self.random_state)
        stages = random.normal(self.canonical_generator.n_biomarker_stages / 2, self.canonical_generator.n_biomarker_stages / 4, self.n_patients)
        stages = np.clip(np.round(stages), 0, self.canonical_generator.n_biomarker_stages - 1).astype(int)
        if self.skewness != 0:
            stages = np.sort(stages) if self.skewness < 0 else np.sort(stages)[::-1]
        return stages

    ### Plotting methods

    def plot_stage_histogram(self) -> None:
        stages = [sample[0] for sample in self.patient_samples]
        plt.hist(stages, bins=self.canonical_generator.n_biomarker_stages, alpha=0.5)
        plt.xlabel('Disease Stage')
        plt.ylabel('Number of Patients')
        plt.title('Patient Stage Distribution')
        plt.show()

    def plot_biomarker_distribution(self, biomarker_index: int, healthy_stage_threshold: float = 2) -> None:
        biomarkers = np.array([sample[1][biomarker_index] for sample in self.patient_samples])
        stages = np.array([sample[0] for sample in self.patient_samples])
        
        healthy = biomarkers[stages <= healthy_stage_threshold * self.canonical_generator.n_biomarker_stages]
        unhealthy = biomarkers[stages > healthy_stage_threshold * self.canonical_generator.n_biomarker_stages]
        
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
