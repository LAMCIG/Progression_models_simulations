import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
# multi_logistic_deriv.m
def multi_logistic_deriv(K, x):
    """calculates the derivative of the logistic function given
    connectivity matrix K and the current state x."""
    dx_dt = (np.eye(len(K)) - np.diag(x)) @ K @ x
    return dx_dt

def random_connectivity_matrix(n, med_frac, source_rate, all_source_connections, seed = 10):
    np.random.RandomState(seed=seed)
    A = np.random.rand(n, n)
    A = A.T @ A
    A = A - np.diag(np.diag(A))
    K = A.copy()
    
    for i in range(n):
        loc_thresh = min(med_frac * np.median(A[i, 1:]), max(A[i, 1:] * 0.99))
        ind = A[i, 1:] < loc_thresh
        ind = np.insert(ind, 0, False)
        K[i, ind] = 0
        K[ind, i] = 0

    S = np.sum(K > 0.0, axis=0)
    for i in range(n):
        if S[i] == 0 or (i == 1 and S[i] < 2):
            if i != 1:
                val, ind = max((val, idx) for (idx, val) in enumerate(A[i, 1:]))
                K[i, ind+1] = val
                K[ind+1, i] = val
            else:
                val, ind = max((val, idx) for (idx, val) in enumerate(A[i, 2:]))
                K[i, ind+2] = val
                K[ind+2, i] = val

    K = K / np.max(K)
    if all_source_connections:
        K[0, :] = source_rate
        K[:, 0] = source_rate
    else:
        K[0, :] = 0.0
        K[:, 0] = 0.0
    K[0, 1] = source_rate
    K[1, 0] = source_rate

    return K

## MAIN
n = 20 # number of biomarkers
step = 0.01 # step size
n_steps = 50
neg_frac = 0.1 # for time before epoch?
t0 = -neg_frac * n_steps * step # start time

# initial conditions
x0 = np.zeros(n)
x0[0] = 1

x = np.zeros((n, int((1 + neg_frac) * n_steps) + 1))
t = np.arange(t0, (n_steps + 1) * step, step)

zero_ind = np.argmin(np.abs(t)) # evil hack

x[:, zero_ind] = x0

K = random_connectivity_matrix(n, 0.5, 0.1, True)

for i in range(1, zero_ind):
    ind = zero_ind - i
    dx_dt = multi_logistic_deriv(K, x[:, ind])
    x[:, ind-1] = np.maximum(0, x[:, ind] - dx_dt * step)

for i in range(zero_ind, n_steps + zero_ind):
    dx_dt = multi_logistic_deriv(K, x[:, i])
    x[:, i+1] = x[:, i] + dx_dt * step

# Assuming you've determined 'x' via manual integration steps
n_time_steps = x.shape[1]  # The number of columns in 'x' matches the number of time steps simulated
t = np.linspace(t0, t0 + step * (n_time_steps - 1), n_time_steps)

plt.plot(t, x.T)
plt.title("euler")
plt.xlabel('time')
plt.ylabel('biomarker progression')
plt.show()


def integrate_system(t, y):
    return multi_logistic_deriv(K, y)

n = 20  # number of biomarkers
step = 0.01  # step size
n_steps = 50
neg_frac = 0.1
t0 = -neg_frac * n_steps * step
tf = n_steps * step  # final time

# initial conditions
x0 = np.zeros(n)
x0[0] = 1

K = random_connectivity_matrix(n, 0.5, 0.1, True)

sol = scipy.integrate.solve_ivp(integrate_system, [t0, tf], x0, method='RK45', max_step=step)

plt.plot(sol.t, sol.y.T)
plt.title("w/ RK45")
plt.xlabel('time')
plt.ylabel('biomarker progression') 
plt.show()
