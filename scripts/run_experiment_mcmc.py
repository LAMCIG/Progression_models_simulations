import argparse
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

from ebm.probability import log_distributions, fit_distributions
from ebm.mcmc import greedy_ascent, mcmc

if __name__=="__main__":
    # EXAMPLE
    # python run_experiment_mcmc.py --file_path /home/kurmukov/SurfAvg_ADNI1_sc.csv 
    #                               --output /data01/bgutman/parkinson_ebm/results/mcmc_results/point_proba/adni/no_prior 
    #                               --connectome_prior /data01/bgutman/parkinson_ebm/log_transition_probabilities_adni.npy
    #                               --point_probability True
    #                               --col_suffix surfavg

    parser = argparse.ArgumentParser(
        description='Runs MCMC optimization for EBM.')
    parser.add_argument('--file_path', help='csv file containing brain regions thickness and target variable')
    parser.add_argument('--output', help='folder to store the results')
    parser.add_argument('--point_probability', 
        help='whether to use cumulative probability or point probability', default=0, type=int)
    parser.add_argument('--connectome_prior', 
        help='path to numpy array with precomputed average connectome prior', default=False)
    parser.add_argument('--col_suffix', help='Features suffix used in the spreadsheet', default='thick')
    parser.add_argument('--stratify', help='Spreadsheet column to stratify by', default=None)
    parser.add_argument('--random_state', help='MCMC random state', default=2020)
    args = parser.parse_args()
    print(type(args.point_probability))
    # file paths
    # '/data01/bgutman/MRI_data/PPMI/EBM_data/corrected_ENIGMA-PD_Mixed_Effects_train_test_split.csv'
    # '/home/kurmukov/ENIGMA-PD-regional.csv'
    # adni
    # '/home/kurmukov/SurfAvg_ADNI1_sc.csv'
    # '/data01/bgutman/MRI_data/ADNI1/Anat_measures/CorticalMeasuresENIGMA_SurfAvg_CROSS_ADNI1_sc.csv'
    # '/data01/bgutman/MRI_data/ADNI1/ADNI_sc_vents840-sorted.csv'

    # Path to file with numpy array with connectivity prior, 
    # `/data01/bgutman/parkinson_ebm/log_transition_probabilities_adni.npy`
        
    # 1. Load data
    data = pd.read_csv(args.file_path, index_col=0)
    cols = [c for c in data.columns if args.col_suffix in c]
    train, test = train_test_split(data, stratify=args.stratify, test_size=0.1, random_state=777)

    try:
        X = train[cols].values
        y = train['Dx'].values
    except KeyError: # check
        print(f'Spreadsheet should contain `{args.col_suffix}` columns and `Dx` columns: {data.columns} were passed.')

    assert X.shape[1] == 68, f'{X.shape}'

    prior = None
    if args.connectome_prior:
        prior = np.load(args.connectome_prior)
    
    # 2. Precomute distributions P(x|E), P(x| not E)
    log_p_e, log_p_not_e = log_distributions(X, y, point_proba=bool(args.point_probability))

    # 3. Run greedy ascent optimization phase
    order, loglike, update_iters = greedy_ascent(log_p_e, log_p_not_e, n_iter=100_000,
                                                 prior=prior, random_state=2020)

    # 4. Save results greedy ascent
    output = Path(args.output)

    np.save(output / 'order_greedy_ascent.npy', np.array(order))
    np.save(output / 'loglike_greedy_ascent.npy', np.array(loglike))
    np.save(output / 'update_iters_greedy_ascent.npy', np.array(update_iters))

    # 5. Run MCMC optimization phase
    orders, loglike, update_iters, probas = mcmc(log_p_e, log_p_not_e,
                                                 order=order, n_iter=1_000_000,
                                                 prior=prior, random_state=2020)

    # 6. Save results MCMC
    np.save(output / 'order_mcmc.npy', np.array(orders))
    np.save(output / 'loglike_mcmc.npy', np.array(loglike))
    np.save(output / 'update_iters_mcmc.npy', np.array(update_iters))
    np.save(output / 'probas_mcmc.npy', np.array(probas))
