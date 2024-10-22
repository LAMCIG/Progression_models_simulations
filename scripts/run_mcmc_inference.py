import sys
import os

sys.path.append('/home/dsemchin/Progression_models_simulations/model_generator')
sys.path.append('/home/dsemchin/Progression_models_simulations/ebm')
sys.path.append('/home/dsemchin/Progression_models_simulations/old_model_generator')

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, kendalltau
from model_generator.model_factory import ModelFactory
from old_model_generator.sample_generator import SampleGenerator
from ebm.probability import log_distributions
from ebm.mcmc import greedy_ascent, mcmc
import itertools
from multiprocessing import Pool

def dprint(var):
    print(var)
    print(type(var))
# created wrappers around each step

## STEP 1: CREATE THE DISEASE MODEL
def create_disease_model(model_name, params):
    print("creating disease model")
    model = ModelFactory.create_model(model_name, **params)
    model.fit()
    # model.plot()
    prior = model.get_connectivity_matrix()
    stage_values = model.transform(X=None)
    print("complete")
    return stage_values, prior

## STEP 2: CREATE THE SAMPLE
def generate_patient_sample(stage_values, n_patients, dist_params, add_noise, noise_std, random_state):
    print("creating disease model")
    print(stage_values.shape)
    print(type(stage_values))
    dprint(n_patients)
    dprint(dist_params)
    dprint(add_noise)
    dprint(random_state)
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
    print("complete")
    return X, y, sample

## STEP 3: RUN EBM
# define how ebm is run
def run_ebm(X, y, prior=None, random_state=1):
    print("running EBM")
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
        'loglike': loglike,
        'log_PE': log_p_e,
        'log_not_PE': log_p_not_e 
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
        'num_iters': result['num_iters']
    }
    
def run_multiple_ebm(X, y, prior, n_trials, csv_filename, n_workers=10):
    output_dir = os.path.dirname(csv_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    args_list = [(X, y, prior, trial) for trial in range(n_trials)]

    with Pool(processes=n_workers) as pool:
        results_list = pool.starmap(run_single_trial, list(args_list))
    
    df = pd.DataFrame(results_list)
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    

def run_simulation(config):
    # get the json file
    disease_model_variations = config['disease_model_variations']

    param_keys = list(disease_model_variations.keys())
    param_values = list(itertools.product(*disease_model_variations.values()))
    
    for param_set in param_values:
        current_params = dict(zip(param_keys, param_set))
        disease_model_params = {**config['disease_model_params'], **current_params}
        stage_values, prior = create_disease_model(
            config['model_name'], disease_model_params
        )
        
        # this was a pretty bad idea in hindsight TODO: remove loop
        for n_patients in config['n_patients']:
            for noise_std in config['noise_levels']:
                X, y, sample = generate_patient_sample(
                    stage_values=stage_values,
                    n_patients=n_patients,
                    dist_params=config['dist_params'],
                    add_noise=config['add_noise'],
                    noise_std=noise_std,
                    random_state=config['random_state']
                )
                
                # construct filenames dynamically based on both noise and disease model parameters
                param_str = '_'.join([f"{key}_{value}" for key, value in current_params.items()])
                no_prior_csv = f"{config['base_csv_name']}_n_{n_patients}_std_{noise_std}_no_prior.csv"
                with_prior_csv = f"{config['base_csv_name']}_{param_str}_n_{n_patients}_std_{noise_std}_with_prior.csv"
                
                n_workers = config.get('n_workers', 10)
                run_multiple_ebm(X=X, y=y, prior=None, n_trials=config['n_trials'], csv_filename=no_prior_csv, n_workers=n_workers)
                run_multiple_ebm(X=X, y=y, prior=prior, n_trials=config['n_trials'], csv_filename=with_prior_csv, n_workers=n_workers)


if __name__ == "__main__":
    config_file = sys.argv[1]
    
    import json
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    run_simulation(config)
# import json
# with open("configs/noise_acp.json", "r") as f:
#     config = json.load(f)
# run_simulation(config)