# doFGPGM.py (ADAPTED)

import numpy as np
import os
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from FGPGM.Experiments.multiBiomarkerLogistic import MultiBiomarkerLogistic as exp
from FGPGM.Kernels.Sigmoid import Sigmoid as kernel
from FGPGM.FGPGM import FGPGM

standardize = np.loadtxt("standardize.csv")
if standardize == 0:
    standardize=False
elif standardize == 1:
    standardize=True
else:
    raise ValueError("Illegal value encountered in standardize.csv.")

time = np.loadtxt("time.csv")
y = np.loadtxt("observations.csv")

experiment = exp()

gammaValue = np.loadtxt("gammaValues.csv")
gammas = gammaValue * np.ones(y.shape[1])

kernels = []
for state in range(y.shape[1]):
    currentKernel = kernel(theta=np.abs(np.random.randn(3)),
                           sigma=np.abs(np.random.randn(1)))
    currentKernel.theta = np.loadtxt(f"hyperparams/theta{state}.csv")
    currentKernel.sigma = np.squeeze(np.loadtxt(f"hyperparams/sigma{state}.csv"))
    kernels.append(currentKernel)

trueTheta = np.loadtxt("trueODEParams.csv")
theta0 = np.abs(np.random.randn(trueTheta.size))

FM = FGPGM(kernels=kernels,
           time=time,
           y=y,
           experiment=experiment,
           nODEParams=trueTheta.size,
           gamma=gammas,
           normalize=True,
           standardize=standardize)

stateStds = np.ones(y.size)*0.001
paramStds = np.ones(theta0.size)*0.15
propStds = np.concatenate([stateStds, paramStds])

newStates, newParams = FM.getFGPGMResults(
    GPPosteriorInit=True,
    blockNegStates=False,
    debug=True,
    theta0=theta0,
    thetaMagnitudes=np.zeros_like(theta0),
    nSamples=300000,
    nBurnin=1000,
    propStds=propStds
)

np.savetxt("optimalParamsFGPGM.csv", newParams)
np.savetxt("optimalStatesFGPGM.csv", newStates)
print("Inferred ODE Params:", newParams)

if not os.path.exists("./plots"):
    os.makedirs("./plots")

noiseObsStd = 0.0
timeDense = np.arange(0, time[-1]+0.01, 0.01)
XInit = np.loadtxt("XInit.csv")

xDenseTrue = experiment.sampleTrajectoryNonUniform(XInit, trueTheta, timeDense, noiseObsStd)[0]
xDenseNew = experiment.sampleTrajectoryNonUniform(XInit, newParams, timeDense, noiseObsStd)[0]

for i in range(xDenseNew.shape[1]):
    plt.figure()
    plt.plot(timeDense, xDenseTrue[:, i], 'k', label='Ground Truth')
    plt.plot(timeDense, xDenseNew[:, i], 'r', label='New Params')
    plt.scatter(time, newStates[:, i], marker='x', c='r', label='New States')
    plt.title(f"State {i}")
    plt.legend()
    plt.savefig(f"./plots/state{i}.png")
    plt.close()
