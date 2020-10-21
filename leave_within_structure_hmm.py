import os
import numpy as np
import matplotlib.pyplot as plt
from structij.lwcv import LWCV
from structij.models.hidden_markov_model import HMM
from structij.utils import genSyntheticDataset, get_indices_in_held_out_fold


# specify problem
D = 2 # data dimension
K = 2 # number of HMM states
T = 1500 # Length of sequence
pct_to_drop = 10 # % of data in the held out fold
contiguous = False # sites dropped in contiguous blocks; else droped via iid sampling.

# generate synthetic data
X, zs, A, pi0, mus = genSyntheticDataset(K=K, T=T, N=1, D=D)

# define and fit model
config_dict = {'K': K, 'precision': None}
model = HMM(X, config_dict)
weights_one = np.ones([T])
# MAP fit
print ("Performing initial model fit!")
opt_params = model.fit(weights_one, init_params=None, num_random_restarts=1,  maxiter=None)
theta_one = opt_params.copy()

# approximate CV
o = get_indices_in_held_out_fold(T=T, pct_to_drop=pct_to_drop, contiguous=contiguous)
lwcv = LWCV(model, theta_one, T, o)
print ("Performing ACV")
params_acv = lwcv.compute_params_acv()

# compare against exact CV
weights = np.ones([T])
weights[o] = 0
approx_cv_loss = model.loss_at_missing_timesteps(weights, params=params_acv)
print ("Performing Exact CV")
params_cv = model.fit(weights, init_params=theta_one)  # exact cv with warm start
exact_cv_loss = model.loss_at_missing_timesteps(weights, params=params_cv)
plt.plot(exact_cv_loss, approx_cv_loss, 'ro', ms=10, alpha=0.5)
plt.plot([0, 5], [0, 5], 'k--', lw=3)
plt.xlabel("Exact CV", fontsize=20)
plt.ylabel("Approximate CV", fontsize=20)

savedir = "sanity_checks/"
if not os.path.exists(savedir):
    os.mkdir(savedir)
savefigpath = savedir + "HMM_ACVvsCV.png"
plt.savefig(savefigpath, dpi=200)
plt.show()