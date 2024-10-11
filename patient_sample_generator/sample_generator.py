from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

class SampleGenerator(BaseEstimator, TransformerMixin):
    """
    Sklearn Transformer for generating patient samples based on disease progression stages.
    
    Parameters:
        stage_values (np.ndarray): 2D array of biomarker values, where rows are biomarkers and columns are stages.
        n_patients (int): Number of artificial patients in your sample.
        distribution (scipy.stats distribution): Distribution to sample patient stages.
        dist_params (dict): Parameters for the chosen distribution.
        add_noise (bool): Adds noise if True.
        noise_std (float): The standard deviation of the noise.
        random_state (int): Set random state for reproducible pseudo-random results.
    
    Attributes:
        patient_samples (pd.DataFrame): DataFrame of generated patient samples with stage and biomarker values.
    """
    def __init__(self, stage_values: np.ndarray, n_patients: int, distribution=norm, dist_params=None, add_noise=True, noise_std=0.1, random_state=None):
        self.n_patients = n_patients
        self.distribution = distribution
        self.dist_params = dist_params if dist_params is not None else {}
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.random_state = random_state
        self.stage_values = None
        self.patient_samples = None

    def fit(self, X, y=None):
        """
        Fit method to set the stage values (disease progression model).
        
        Parameters:
            X (np.ndarray): 2D array where rows are biomarkers and columns are disease stages.
        """
        self.stage_values = X
        return self

    def transform(self, X=None):
        """
        Transform method to generate patient samples based on the stage values.
        
        Returns:
            pd.DataFrame: DataFrame of generated patient samples, including stages and biomarker values.
        """
        if self.stage_values is None:
            raise ValueError("The model must be fitted with stage_values before calling transform.")
        
        self.patient_samples = self._generate_patient_samples()
        return self.patient_samples

    def _generate_patient_samples(self) -> pd.DataFrame:
        # fix randomness and generate sample stages
        random = np.random.RandomState(self.random_state)
        sample_stages = self._generate_stages()
        sample_data = []
        
        for stage in sample_stages:
            biomarker_values = self.stage_values[:, stage]  # get biomarker values for the stage
            if self.add_noise:
                noise = random.normal(0, self.noise_std, biomarker_values.shape)
                biomarker_values = np.clip(biomarker_values + noise, 0, 1)  # ensure values stay in [0,1]
            
            # append stage and biomarker values for each patient
            sample_data.append({'stage': stage, 'biomarker_values': biomarker_values})
        
        df = pd.DataFrame(sample_data)
        df['target'] = (df['stage'] > 2).astype(int)  # xample: stages > 2 are considered 'unhealthy'
        
        return df

    def _generate_stages(self) -> np.ndarray:
        """
        Generate random patient stages based on the chosen distribution.
        
        Returns:
            np.ndarray: Array of stages (integers) for each patient.
        """
        n_stages = self.stage_values.shape[1]
        stages = self.distribution.rvs(size=self.n_patients, **self.dist_params)
        stages = np.clip(np.round(stages), 0, n_stages - 1).astype(int)
        stages = np.sort(stages)  # Sort for better visualization
        return stages

    def get_X(self) -> np.ndarray:
        """
        Return the biomarker values (X) for the generated patient samples.
        """
        X = np.vstack(self.patient_samples['biomarker_values'].values)
        return X

    def get_y(self) -> np.ndarray:
        """
        Return the target values (y) for the generated patient samples.
        """
        y = self.patient_samples['target'].values
        return y

    def plot_stage_histogram(self):
        """Plots histogram of patient stages."""
        sns.histplot(self.patient_samples['stage'], bins=self.stage_values.shape[1], kde=False, color='skyblue', edgecolor='black')
        plt.xlabel('Disease Stage')
        plt.ylabel('Number of Patients')
        plt.title('Distribution of Patient Stages')
        plt.show()
