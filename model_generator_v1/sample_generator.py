import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform, erlang, pareto

class SampleGenerator:
    """
    Sample generator takes in stage values and parameters for generating a patient sample.
    
    Parameters:
        stage_values (np.ndarray): 2D array of biomarker values, where rows are biomarkers and columns are stages.
        n_patients (int): Number of artificial patients in your sample.
        distribution (scipy.stats distribution): Distribution to sample patient stages.
        dist_params (dict): Parameters for the chosen distribution.
        add_noise (bool): Adds noise if True.
        noise_std (float): The standard deviation of the noise.
        random_state (int): Set random state for reproducible pseudo-random results.
    
    Attributes:
        patient_samples (list): List of tuples with stage and biomarker values for each patient. For each element (patient_stage: int, biomarker_values: np.ndarray)
    """
    def __init__(self, stage_values: np.ndarray, n_patients: int, distribution=norm, dist_params=None, add_noise: bool = True, noise_std: float = 0.1, random_state: int = None):
        self.stage_values = stage_values
        self.n_patients = n_patients
        self.distribution = distribution
        self.dist_params = dist_params if dist_params is not None else {}
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.random_state = random_state
        
        ## Attribute
        self.patient_samples = self._generate_patient_samples()

    def _generate_patient_samples(self) -> pd.DataFrame:
        # fix randomness and initialize array size of n_patients
        random = np.random.RandomState(self.random_state)
        sample_stages = self._generate_stages()
        
        sample_data = []
        
        for stage in sample_stages:
            biomarker_values = self.stage_values[:, stage] # get all biomarker values at specified stage
            if self.add_noise:
                noise = random.normal(0, self.noise_std, biomarker_values.shape)
                biomarker_values = np.clip(biomarker_values + noise, 0, 1)  # ensures values are in the range [0,1]
            # Append each row of data for the patient: stage, biomarker values, target (healthy/unhealthy)
            sample_data.append({'stage': stage, 'biomarker_values': biomarker_values})
        
        df = pd.DataFrame(sample_data)
        
        df['target'] = (df['stage'] > 2).astype(int)  # Assume 'healthy' is stage <= 2, stage 0 and 1 is healthy
        
        return df
    
    def _generate_stages(self) -> np.ndarray:
        """
        Returns a sorted ndarray (ascending) of statistically distibuted ints on interval [0, n_biomarkers - 1]
        """
        n_stages = self.stage_values.shape[1] # recall: columns = stages
        stages = self.distribution.rvs(size=self.n_patients, **self.dist_params)
        
        # clip and round stages ensures they fall within the correct range
        stages = np.clip(np.round(stages), 0, n_stages - 1).astype(int)
        stages = np.sort(stages)
        
        return stages

    def get_sample(self) -> pd.DataFrame:
        return self.patient_samples
    
    def get_X(self) -> np.ndarray:
        X = np.vstack(self.patient_samples['biomarker_values'].values)
        return X

    def get_y(self) -> np.ndarray:
        y = self.patient_samples['target'].values
        return y
    
    #%% Plotting methods
    def plot_stage_histogram(self) -> None:
        """Plots histogram of patient stages."""
        sns.histplot(self.patient_samples['stage'], bins=self.stage_values.shape[1], kde=False, color='skyblue', edgecolor='black')
        plt.xlabel('Disease Stage')
        plt.ylabel('Number of Patients')
        plt.title('Distribution of Patient Stages')
        plt.show()

    def plot_biomarker_distribution(self, stage: int) -> None:
        """Plots distribution of biomarker values for patients at a specific stage."""
        # Extract the threshold value from the diagonal of stage_values
        threshold_value = self.stage_values[stage, stage]
        
        # Extract biomarker values and stages
        biomarker_values = np.hstack(self.patient_samples['biomarker_values'])
        stages = np.repeat(self.patient_samples['stage'], repeats=len(self.stage_values))
        df = pd.DataFrame({'stage': stages, 'biomarker_value': biomarker_values})
        
        sns.histplot(df[df['stage'] == stage]['biomarker_value'], bins=20, kde=True, label='All', color='gray', alpha=0.2)
        sns.histplot(df[(df['stage'] == stage) & (df['biomarker_value'] <= threshold_value)]['biomarker_value'], bins=20, kde=True, color='green', label='Healthy', alpha=0.5)
        sns.histplot(df[(df['stage'] == stage) & (df['biomarker_value'] > threshold_value)]['biomarker_value'], bins=20, kde=True, color='red', label='Unhealthy', alpha=0.5)

        plt.xlim([0,1])
        plt.xlabel('Biomarker Value')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Biomarker Values for Stage {stage}')
        plt.legend()
        plt.show()

    def plot_patient_biomarkers(self, patient_index: int) -> None:
        """Plots biomarker values for a specific patient."""
        patient_sample = self.patient_samples.iloc[patient_index]
        stage = patient_sample['stage']
        biomarker_values = patient_sample['biomarker_values']
        
    def plot_vertical_boxplots(self, stage: int) -> None:
        """Creates vertical box plots for each biomarker's distribution at a specified stage."""
        stage_data = self.patient_samples[self.patient_samples['stage'] == stage]
               
        biomarker_matrix = np.vstack(stage_data['biomarker_values'].values)
        df_biomarkers = pd.DataFrame(biomarker_matrix, columns=[f'biomarker {i+1}' for i in range(biomarker_matrix.shape[1])])
        
        sns.boxplot(data=df_biomarkers, orient='v', palette='Set2')
        plt.title(f'Biomarker Value Distributions at Stage {stage}')
        plt.xlabel('Biomarker')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.show()