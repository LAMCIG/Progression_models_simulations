import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm # for controlling size of each patient stage sample size
class SampleGenerator:
    def __init__(self, canonical_generator, n_patients, add_noise=False, noise_std=0.1, random_state=None, skewness=0):
        self.canonical_generator = canonical_generator
        self.n_patients = n_patients
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.random_state = np.random.RandomState(random_state)
        self.skewness = skewness
        self.patient_samples = self._generate_patient_samples()
    
    def _generate_patient_samples(self):
        patients = []
        stages = self._generate_patient_stages()
        for stage in stages:
            biomarkers = self._generate_patient_biomarkers(stage)
            patients.append((stage, biomarkers))
        return patients
    
    def _generate_patient_stages(self):
        """
        Generates patient stages based on a skewed normal distribution.
        """
        mean = self.canonical_generator.n_stages / 2
        std_dev = self.canonical_generator.n_stages / 4
        
        # skewnorm used to generate a skewed normal distribution
        skew_param = self.skewness  # recall: positive values skew right, negative values skew left
        skewed_stages = skewnorm.rvs(a=skew_param, loc=mean, scale=std_dev, size=self.n_patients, random_state=self.random_state)
        stages = np.clip(skewed_stages, 0, self.canonical_generator.n_stages - 1).astype(int)
        
        return stages

    def _generate_patient_biomarkers(self, stage):
        biomarkers = []
        for biomarker in range(self.canonical_generator.n_biomarkers):
            value = self.canonical_generator.model_predict(stage, biomarker)
            if self.add_noise:
                value += self.random_state.normal(0, self.noise_std)
                value = np.clip(value, 0, 1)
            biomarkers.append(value)
        return biomarkers
    
    def plot_stage_histogram(self):
        stages = [sample[0] for sample in self.patient_samples]
        plt.hist(stages, bins=range(self.canonical_generator.n_stages + 1), edgecolor='k')
        plt.xlabel('Disease Stage')
        plt.ylabel('Number of Patients')
        plt.title('Histogram of Patient Stages')
        plt.show()

    def plot_biomarker_distribution(self, biomarker_index, threshold=0.5):
        biomarkers = np.array([sample[1] for sample in self.patient_samples])
        stages = np.array([sample[0] for sample in self.patient_samples])
        biomarker_values = biomarkers[:, biomarker_index]

        healthy = biomarker_values[stages < threshold * self.canonical_generator.n_stages]
        unhealthy = biomarker_values[stages >= threshold * self.canonical_generator.n_stages]

        bins = plt.hist(biomarker_values, alpha=0.1, label='all', bins=20)
        plt.hist(healthy, bins=bins[1], alpha=0.5, label='healthy')
        plt.hist(unhealthy, bins=bins[1], alpha=0.5, label='unhealthy')
        plt.xlabel('Biomarker Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Biomarker {biomarker_index + 1}')
        plt.legend()
        plt.show()
