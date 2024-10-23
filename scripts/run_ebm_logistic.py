import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, kendalltau
from model_generator.model_factory import ModelFactory
from old_model_generator.sample_generator import SampleGenerator
from ebm.probability import log_distributions
from ebm.mcmc import greedy_ascent, mcmc
from multiprocessing import Pool

## DISEASE MODEL CREATION
params = {
    'n_stages': 10,  # Number of biomarkers
    'step': 0.1,
    'n_steps': 100,
    'start_time': 0,
    'end_time': 100,
    'steps': 100,
    'connectivity_matrix_type': 'random_connectivity',  # Connectivity matrix type
    'convergence_threshold': 1e-4,
    'flip_v': True
}

model_type = 'logistic'
model = ModelFactory.create_model(model_type=model_type, **params)
model.fit()
stage_values = model.transform(X=None)  # Only works when X is explicitly defined as None!
prior = model.get_connectivity_matrix()

## STEP 2: CREATE THE SAMPLE
def generate_patient_sample(stage_values, n_patients, dist_params, add_noise, noise_std, random_state):
    sample = SampleGenerator(
        stage_values=stage_values,
        n_patients=n_patients,
        distribution=norm,
        dist_params=dist_params,
        add_noise=add_noise,
        noise_std=noise_std,
        random_state=random_state
    )
    X = sample.get_X()
    y = sample.get_y()
    return X, y, sample

## STEP 3: RUN EBM
def run_ebm(X, y, prior=None, random_state=1):
    log_p_e, log_p_not_e = log_distributions(X, y, point_proba=False)
    rng = np.random.RandomState(random_state)
    ideal_order = np.arange(X.shape[1])
    starting_order = rng.choice(ideal_order, len(ideal_order), replace=False)
    starting_order_copy = starting_order.copy()

    order, loglike, update_iters = greedy_ascent(log_p_e, log_p_not_e, 
                                                 n_iter=10_000, order=starting_order,
                                                 prior=prior, random_state=random_state)
    
    orders, loglike, update_iters, probas = mcmc(log_p_e, log_p_not_e, 
                                                 order=order, n_iter=500_000, 
                                                 prior=prior, random_state=random_state)
    
    best_order = orders[np.argmax(loglike)].copy() if len(orders) != 0 else order.copy()

    return {
        'starting_order': starting_order_copy,
        'greedy_order': order,
        'best_order': best_order,
        'spearmanr': [spearmanr(ideal_order, starting_order_copy)[0],
                      spearmanr(ideal_order, order)[0],
                      spearmanr(ideal_order, best_order)[0]],
        'kendalltau': [kendalltau(ideal_order, starting_order_copy)[0],
                       kendalltau(ideal_order, order)[0],
                       kendalltau(ideal_order, best_order)[0]],
        'num_iters': len(orders) if orders is not None else 0,
        'loglike': loglike
    }

# single wrapper for parallel trials
def run_single_trial(X, y, prior, trial):
    result = run_ebm(X, y, prior=prior, random_state=trial)
    return {
        'run': trial,
        'starting_order': result['starting_order'].tolist(),
        'greedy_order': result['greedy_order'].tolist(),
        'best_order': result['best_order'].tolist(),
        'starting_spearmanr': result['spearmanr'][0],
        'greedy_spearmanr': result['spearmanr'][1],
        'best_spearmanr': result['spearmanr'][2],
        'starting_kendalltau': result['kendalltau'][0],
        'greedy_kendalltau': result['kendalltau'][1],
        'best_kendalltau': result['kendalltau'][2],
        'num_iters': result['num_iters'],
        'loglike': result['loglike']
    }

# Run all 30 trials for one combination
def run_multiple_ebm(X, y, prior, n_trials, csv_filename, n_workers=15):
    output_dir = os.path.dirname(csv_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args_list = [(X, y, prior, trial) for trial in range(n_trials)]

    with Pool(processes=n_workers) as pool:
        results_list = pool.starmap(run_single_trial, list(args_list))

    df = pd.DataFrame(results_list)
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

def run_simulation():
    sample_sizes = [100, 250, 500, 1000]
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]

    n_trials = 30  # Always running 30 trials in each file
    n_workers = 15  # Adjust based on your system's resources

    for n_patients in sample_sizes:
        for noise_std in noise_levels:
            X, y, sample = generate_patient_sample(
                stage_values=stage_values,
                n_patients=n_patients,
                dist_params={'loc': 2.5, 'scale': 3.5, 'random_state': 10},
                add_noise=True,
                noise_std=noise_std,
                random_state=10
            )

            # Create filenames for the CSVs
            no_prior_csv = f"results/{model_type}_n_{n_patients}_std_{noise_std}_no_prior.csv"
            with_prior_csv = f"results/{model_type}_n_{n_patients}_std_{noise_std}_with_prior.csv"

            # Run 30 trials without prior
            run_multiple_ebm(X=X, y=y, prior=None, n_trials=n_trials, csv_filename=no_prior_csv, n_workers=n_workers)

            # Run 30 trials with prior
            run_multiple_ebm(X=X, y=y, prior=prior, n_trials=n_trials, csv_filename=with_prior_csv, n_workers=n_workers)

if __name__ == "__main__":
    run_simulation()
