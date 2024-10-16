import sys
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, kendalltau
from model_generator.model_factory import ModelFactory
from patient_sample_generator.sample_generator import SampleGenerator
from ebm.probability import log_distributions
from ebm.mcmc import greedy_ascent, mcmc
import itertools
import os
from multiprocessing import Pool


## STEP 1: CREATE THE DISEASE MODEL
def create_disease_model(model_name, params):
    model = ModelFactory.create_model(model_name, **params)
    model.fit()
    # model.plot()
    prior = model.get_connectivity_matrix()
    stage_values = model.transform(X=None)
    return stage_values, prior

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
    # sample.plot_stage_histogram()
    X = sample.get_X()
    y = sample.get_y()
    return X, y, sample

## STEP 3: RUN EBM
# define how ebm is run
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
        'num_iters': len(orders) if orders is not None else 0
    }

# single wrapper for parallel trials
def run_single_trial(args):
    X, y, prior, trial = args
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
        'num_iters': result['num_iters']
    }

# parallelized version of run_multiple_ebm
def run_multiple_ebm(X, y, prior, n_trials, csv_filename, n_workers=4):
    output_dir = os.path.dirname(csv_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # args list for all trials
    args_list = [(X, y, prior, trial) for trial in range(n_trials)]
    
    # execute trials in parallel
    with Pool(processes=n_workers) as pool:
        results_list = pool.map(run_single_trial, args_list)
    
    # save results to CSV
    df = pd.DataFrame(results_list)
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    
    df = pd.DataFrame(results_list)
    df.to_csv(csv_filename, index=False)
    

def run_simulation(config):
    # Extract the varying disease model parameters
    disease_model_variations = config['disease_model_variations']

    # Generate all combinations of the varying disease model parameters
    param_keys = list(disease_model_variations.keys())
    param_values = list(itertools.product(*disease_model_variations.values()))
    
    # Iterate over each combination of disease model parameters
    for param_set in param_values:
        current_params = dict(zip(param_keys, param_set))
        
        # Merge current parameter set with the fixed disease model params
        disease_model_params = {**config['disease_model_params'], **current_params}
        
        # Step 1: Create the disease model
        stage_values, prior = create_disease_model(
            config['model_name'], disease_model_params
        )
        
        # Step 2: Iterate over noise levels and run simulations for each
        for noise_std in config['noise_levels']:
            # Generate patient sample with current noise level
            X, y, sample = generate_patient_sample(
                stage_values=stage_values,
                n_patients=config['n_patients'],
                dist_params=config['dist_params'],
                add_noise=config['add_noise'],
                noise_std=noise_std,
                random_state=config['random_state']
            )
            
            # Construct filenames dynamically based on both noise and disease model parameters
            param_str = '_'.join([f"{key}_{value}" for key, value in current_params.items()])
            no_prior_csv = f"{config['base_csv_name']}_{param_str}_noise_{noise_std}_no_prior.csv"
            with_prior_csv = f"{config['base_csv_name']}_{param_str}_noise_{noise_std}_with_prior.csv"
            
            # Get the number of workers from the config
            n_workers = config.get('n_workers', 5)
            
            # Step 3: Run inference in parallel
            print(f"Running without prior for noise {noise_std} and params {current_params}")
            run_multiple_ebm(X=X, y=y, prior=None, n_trials=config['n_trials'], csv_filename=no_prior_csv, n_workers=n_workers)
            
            if config['use_prior']:
                print(f"Running with prior for noise {noise_std} and params {current_params}")
                run_multiple_ebm(X=X, y=y, prior=prior, n_trials=config['n_trials'], csv_filename=with_prior_csv, n_workers=n_workers)


if __name__ == "__main__":
    config_file = sys.argv[1]
    
    import json
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    run_simulation(config)