import numpy as np

class SampleGenerator:
    def __init__(self, canonical_generator, n_patients, add_noise=False, noise_std=0.1, random_state=None):
        self.canonical_generator = canonical_generator
        self.n_patients = n_patients
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.random_state = np.random.RandomState(random_state)
        self.patient_samples = self._generate_patient_samples()
    
    def _generate_patient_samples(self):
        patients = []
        for _ in range(self.n_patients):
            stage = self.random_state.randint(0, self.canonical_generator.n_stages)
            biomarkers = self._generate_patient_biomarkers(stage)
            patients.append((stage, biomarkers))
        return patients
    
    def _generate_patient_biomarkers(self, stage):
        biomarkers = []
        for biomarker in range(self.canonical_generator.n_biomarkers):
            value = self.canonical_generator.model_predict(stage, biomarker)
            if self.add_noise:
                value += self.random_state.normal(0, self.noise_std)
                value = np.clip(value, 0, 1)
            biomarkers.append(value)
        return biomarkers
