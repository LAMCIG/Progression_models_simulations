# TODO: Capture warnings

import numpy as np
import pytest
from simulation_class.canonical_generator import CanonicalGenerator
from simulation_class.sample_generator import SampleGenerator
from simulation_class.disease_progression_analyzer import DiseaseProgressionAnalyzer
from simulation_class.EBMAnalyzer import EBMAnalyzer

@pytest.fixture # call 'pytest' at the root
def setup_sigmoid_model():
    n_biomarkers = 10
    n_stages = 10
    model_type = 'sigmoid'
    biomarkers_params_sigmoid = {
        0: {'s': 2.5, 'c': 2.0},
        1: {'s': 2.5, 'c': 2.5},
        2: {'s': 2.5, 'c': 3.0},
        3: {'s': 2.5, 'c': 3.5},
        4: {'s': 2.5, 'c': 4.0},
        5: {'s': 2.5, 'c': 4.5},
        6: {'s': 2.5, 'c': 5.0},
        7: {'s': 2.5, 'c': 5.5},
        8: {'s': 2.5, 'c': 6.0},
        9: {'s': 2.5, 'c': 6.5},
    }
    canonical_generator = CanonicalGenerator(n_biomarkers, n_stages, model_type, biomarkers_params=biomarkers_params_sigmoid)
    return canonical_generator

def test_canonical_generator_sigmoid(setup_sigmoid_model):
    canonical_generator = setup_sigmoid_model
    assert len(canonical_generator.model) == 10, "There should be 10 biomarkers in the model"
    assert all(len(canonical_generator.model[biomarker]) == 10 for biomarker in canonical_generator.model), "Each biomarker should have 10 stages"

def test_sample_generator(setup_sigmoid_model):
    canonical_generator = setup_sigmoid_model
    n_patients = 100
    sample_generator = SampleGenerator(canonical_generator, n_patients, add_noise=False, noise_std=0, random_state=2, skewness=-1)
    assert len(sample_generator.patient_samples) == 100, "There should be 100 patient samples"

def test_ebm_analyzer(setup_sigmoid_model):
    canonical_generator = setup_sigmoid_model
    n_patients = 100
    sample_generator = SampleGenerator(canonical_generator, n_patients, add_noise=False, noise_std=0, random_state=2, skewness=-1)
    patient_samples = sample_generator.patient_samples
    analyzer = DiseaseProgressionAnalyzer(patient_samples)
    orders, rho, loglike, update_iters, probas = analyzer.run_analysis('ebm')
    assert len(orders) > 0, "There should be at least one order generated"
    assert -1 <= rho <= 1, "Spearman's rho should be between -1 and 1"

if __name__ == "__main__":
    pytest.main()
