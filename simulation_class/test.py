from biomarker_simulation import BiomarkerSimulation
from biomarker_utils import generate_transition_matrix, initialize_biomarkers
from mcmc_analysis import MCMCAnalysis
# Common params
n_patients_stage = {
    1: 200, 2: 250, 3: 200, 4: 70, 5: 80,
    6: 60, 7: 50, 8: 40, 9: 40, 10: 30,
}

### Sigmoid Params
biomarkers_params_sigmoid = {
    0: {'s': 6, 'c': 3},
    1: {'s': 12, 'c': 12},
    2: {'s': 17, 'c': 10},
    3: {'s': 23, 'c': 11},
    4: {'s': 28, 'c': 2},
    5: {'s': 34, 'c': 12},
    6: {'s': 40, 'c': 10},
    7: {'s': 45, 'c': 8},
    8: {'s': 51, 'c': 6},
    9: {'s': 57, 'c': 5},
    10: {'s': 62, 'c': 2}
}

### Transition Matrix Params
num_biomarkers = 11
A = generate_transition_matrix(size=num_biomarkers, coeff=1e-1)
y_init = initialize_biomarkers(num_biomarkers, init_value=0.9)

biomarkers_params_transition = {
    'transition_matrix': A[1:, 1:],  # Assume first row/column are removed
    'y_init': y_init[1:]  # Assume first value is removed
}

# Instantiate simulators for each method
simulator_sigmoid = BiomarkerSimulation(n_patients_stage, biomarkers_params_sigmoid, method='sigmoid_inv')
simulator_transition = BiomarkerSimulation(n_patients_stage, biomarkers_params_transition, method='transition_matrix')

# Run simulations
X_sigmoid, y_sigmoid, stages_sigmoid = simulator_sigmoid.simulate()
X_transition, y_transition, stages_transition = simulator_transition.simulate()

print("Sigmoidal:")
print("X shape:", X_sigmoid.shape)
print("Y shape:", y_sigmoid.shape)
print("Stages shape:", stages_sigmoid.shape)

print("\nTransition Matrix:")
print("X shape:", X_transition.shape)
print("Y shape:", y_transition.shape)
print("Stages shape:", stages_transition.shape)


# mcmc_analysis = MCMCAnalysis(simulator_transition)
# starting_order = np.arange(simulator.X.shape[1]) 
# np.random.shuffle(starting_order)

# order, loglike, _ = mcmc_analysis.run_greedy_ascent(starting_order, random_state=2020, prior=None)  # Example with no prior
# orders, _, _, _ = mcmc_analysis.run_mcmc(order, random_state=2020, prior=None)


### plotting utitlity for later

import matplotlib.pyplot as plt

def plot_biomarker_distribution(X, biomarker, method):
    plt.figure(figsize=(10, 5))
    plt.hist(X[:, biomarker], alpha=0.5, bins=30, label=f'Biomarker {biomarker}')
    plt.title(method)
    plt.xlabel('biomarker value')
    plt.ylabel('frequency')
    plt.legend()
    plt.show()

plot_biomarker_distribution(X_sigmoid, 3,"Sigmoid")
plot_biomarker_distribution(X_transition, 3,"Transition Matrix")
