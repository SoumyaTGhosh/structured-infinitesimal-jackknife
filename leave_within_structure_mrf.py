"""
Minimal example running ACV on a Potts model analyzing 
Philly crime data. Currently, the leave-one-out folds are 
sequentially processed but using parallel computing, like 
Pool from multiprocessing, is straightforward. 
"""

# standard libraries 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

# our implementation 
from src.lwcv import LWCV
from src.models.potts import Potts
from src.utils import extract_folds

# load data 
filepath = "./data/subset2-ver2.npz"
data, N = extract_folds(filepath)
beta = 1.0

# fit with all data
config_dict = {"beta": beta, "display": False, "heuristic": "MinFill"}
model = Potts(data['full'], config_dict)
opt_params = model.fit(r_init=None, display=False)
theta_ones = opt_params.copy()

ACVpredictives = np.zeros(N)
exactpredictives = np.zeros(N)
# leave-one-out fits
for i in tqdm(range(N)):
    lwcv = LWCV(model, theta_ones, N, i)
    params_acv = lwcv.compute_params_acv()
    """
    print(type(i))
    print(type(params_acv))
    print(ACVpredictives[i])
    """
    ACVpredictives[i], _ = model.loo_predictive(missing_site=i, params=params_acv, display=False)

    loo_model = Potts(data[i], config_dict)
    params_exact = loo_model.fit(r_init=theta_ones,display=False)
    exactpredictives[i], _ = model.loo_predictive(missing_site=i, params=params_exact, display=False)

# compare CV with ACV
savefigpath = "sanity_checks/subset2_ver2_ACVvsCV.png"
plt.plot(-exactpredictives, -ACVpredictives, 'ro', ms=10, alpha=0.5)
cap = max(max(-exactpredictives), max(-ACVpredictives))
plt.plot([0, cap], [0, cap], 'k--', lw=3)
plt.show()
plt.savefig(savefigpath)