import numpy as np
from tqdm import tqdm
from .likelihood import EventProbabilities

# TODO: combine greedy_ascent and mcmc into a single func

def greedy_ascent(log_p_E: np.ndarray, log_p_not_E: np.ndarray,
                  order: np.ndarray=None, n_iter: int=10_000, prior=None, random_state: int=None):
    """Performs greedy ascent optimization phase."""
    
    if order is None:
        order = np.arange(log_p_E.shape[1])
    if random_state is None or isinstance(random_state, int):
        random = np.random.RandomState(random_state)
    else:
        raise TypeError
        
    indices = np.arange(len(order))
    model = EventProbabilities(log_p_E, log_p_not_E)
    loglike, update_iters = [], []
    old_loglike = model.compute_total_likelihood(order, prior=prior)
        
    for i in tqdm(range(n_iter)):
        random.shuffle(indices)
        a, b = indices[0], indices[1]
        order[a], order[b] = order[b], order[a]
        new_loglike = model.compute_total_likelihood(order, prior=prior)
        if new_loglike > old_loglike:
            old_loglike = new_loglike
            loglike.append(old_loglike)
            update_iters.append(i)
        else:
            order[a], order[b] = order[b], order[a]
    return order, loglike, update_iters


def mcmc(log_p_E: np.ndarray, log_p_not_E: np.ndarray,
                  order: np.ndarray=None, n_iter: int=100_000, prior=None, random_state: int=None):
    """Performs MCMC optimization phase."""
    
    if order is None:
        order = np.arange(log_p_E.shape[1])
    if random_state is None or isinstance(random_state, int):
        random = np.random.RandomState(random_state)
    else:
        raise TypeError
        
    indices = np.arange(len(order))
    model = EventProbabilities(log_p_E, log_p_not_E)
    orders, loglike, probas, update_iters = [], [], [], []
    old_loglike = model.compute_total_likelihood(order, prior=prior)
        
    for i in tqdm(range(n_iter)):
        random.shuffle(indices)
        a, b = indices[0], indices[1]
        order[a], order[b] = order[b], order[a]
        new_loglike = model.compute_total_likelihood(order, prior=prior)
        p = np.exp(new_loglike - old_loglike)
        if p > random.random_sample(): # TODO: check probas validity
            old_loglike = new_loglike
            loglike.append(old_loglike)
            update_iters.append(i)
            orders.append(order.copy())     
            probas.append(p)
        else:
            order[a], order[b] = order[b], order[a]
    return orders, loglike, update_iters, probas


def get_optimal_order(orders: list):
    """Computes optimal order from list of orders obtained over MCMC runs."""
    # TODO: number of regions is hardcoded
    # How many time region `r` is on position `i`
    n_stages = len(orders[0])
    order_map = np.zeros((n_stages,n_stages))
    for i in range(n_stages):
        region, freq = np.unique(orders[:, i], return_counts=True)
        for r, f in zip(region, freq):
            order_map[r, i] = f
            
    # Optimal order
    best_order = []
    for i in range(n_stages):
        candidate_regions = np.argsort(order_map[:, i])[::-1]
        for reg in candidate_regions:
            if reg not in best_order:
                best_order.append(reg)
                break
    best_order = np.array(best_order)  
    return order_map, best_order

