# -*- coding: utf-8 -*-
"""
Creates and saves all the hyperparameters needed by doFGPGM.py.
Should only be run after createExperiments.py.
"""
import numpy as np
import os
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from FGPGM.Kernels.Sigmoid import Sigmoid

standardize = True  
np.savetxt("standardize.csv", np.array([int(standardize)]).reshape([1, 1]))

kernelIter = 100  
gammaValue = 1e-4
np.savetxt("gammaValues.csv", np.array([gammaValue]).reshape([-1, 1]))
y = np.loadtxt("observations.csv")
time = np.loadtxt("time.csv")

if y.ndim == 1:
    y = y[:, np.newaxis]

num_states = y.shape[1]

for state in range(num_states):
    currentKernel = Sigmoid(theta=np.abs(np.random.randn(3)),
                            sigma=np.abs(np.random.randn(1)))
    
    currentKernel.learnHyperparams(
        currentKernel.theta,
        currentKernel.sigma,
        y[:, state],
        time,
        normalize=True,
        standardize=standardize,
        newNugget=1e-4,
        anneal=False,
        basinIter=kernelIter
    )
    
    if not os.path.exists("./hyperparams"):
        os.makedirs("./hyperparams")
    
    np.savetxt("hyperparams/theta{}.csv".format(state), currentKernel.theta)
    np.savetxt("hyperparams/sigma{}.csv".format(state), np.asarray(currentKernel.sigma).reshape([1, 1]))

print("Hyperparameter optimization completed and saved!!!")
