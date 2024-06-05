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

    def plot_biomarker_distribution(self, biomarker_index, healthy_stage_threshold=3):
        """
        Plots the distribution of a specific biomarker for healthy and unhealthy patients.
        
        Parameters:
        biomarker_index (int): Index of the biomarker to plot.
        healthy_stage_threshold (int): Stage threshold below which patients are considered healthy.
        """
        # extract biomarker values and stages from patient samples
        biomarkers = np.array([sample[1] for sample in self.patient_samples])
        stages = np.array([sample[0] for sample in self.patient_samples])
        biomarker_values = biomarkers[:, biomarker_index]

        # define health status based on stage: stages <= healthy_stage_threshold are considered healthy, > healthy_stage_threshold are diseased
        healthy = biomarker_values[stages <= healthy_stage_threshold]
        unhealthy = biomarker_values[stages > healthy_stage_threshold]

        # plot histogram for all, healthy, and unhealthy patients
        bins = plt.hist(biomarker_values, alpha=0.1, label='all', bins=20)
        plt.hist(healthy, bins=bins[1], alpha=0.5, label='healthy')
        plt.hist(unhealthy, bins=bins[1], alpha=0.5, label='unhealthy')
        plt.xlabel('Biomarker Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Biomarker {biomarker_index + 1}')
        plt.legend()
        plt.show()

        
    def plot_patient_biomarkers(self, patient_index):
        """
        Plots the biomarker progression for all stages and superimposes the biomarker values of a specific patient.

        Parameters:
        patient_index (int): The index of the patient in the sample whose biomarker values will be plotted.
        """
        if patient_index >= self.n_patients:
            raise ValueError(f"Patient index {patient_index} is out of range. There are only {self.n_patients} patients.")
        
        patient_stage, patient_biomarkers = self.patient_samples[patient_index]

        for biomarker in range(self.canonical_generator.n_biomarkers):
            plt.plot(np.arange(self.canonical_generator.n_stages), 
                     [self.canonical_generator.model_predict(stage, biomarker) for stage in range(self.canonical_generator.n_stages)], 
                     label=f'Biomarker {biomarker + 1}', alpha=0.5)
        
        plt.scatter([patient_stage] * self.canonical_generator.n_biomarkers, 
                    patient_biomarkers, 
                    color='red', 
                    zorder=5, 
                    label=f'Patient {patient_index} Biomarkers')
        
        plt.xlabel('Disease Stage')
        plt.ylabel('Biomarker Value')
        plt.title('Biomarker Progression with Patient Biomarkers')
        plt.legend()
        plt.show()
