# Run experiment on crime_tracts or subset0. The current code
# is serial (one leave-site-out experiment after another) but 
# parallelizing is straightforward.

import numpy as np
from models.potts import Potts
from models.mrf_utils import extract_folds
import matplotlib.pyplot as plt 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--file_name', type=str, default='subset0.npz')
    args = parser.parse_args()
    
    path = args.file_name
    data, N = extract_folds('data/'+ path)
    beta = args.beta
    
    heuristic = "MinFill" # models.utils EliminationOrdering supports MinFill and MinNeighbors

    results_dict = {}
    # full-data fit
    fullMRF = Potts(data['full'], beta, heuristic, display=True) # report some basic statistics
    full_data_r, times = fullMRF.full_data_fit(r_init=None,get_visuals=True,compute_sensitivity=True)
    results_dict['full'] = full_data_r

    IJtime = times['fit']+times['sensitivity']
    exacttime = times['fit']

    zeroth_errors = np.zeros(N)
    IJpredictives = np.zeros(N)
    exactpredictives = np.zeros(N)
    first_errors = np.zeros(N)
    results_dict['numsites'] = N
    for i in range(N):
        results_dict[str(i)] = np.zeros((2,2))
        # IJ approximation of refits
        weights = np.ones(N)
        weights[i] = 0.0
        IJ, refit_time = fullMRF.retrain_with_weights(weights)
        IJpredictives[i], t = fullMRF.loo_predictive(missing_site=i, params=IJ, display=True)
        results_dict[str(i)][0,:] = IJ
        IJtime += refit_time + t
        # exact leave-one-out refits, using full data solution as initialization
        looMRF = Potts(data[i],beta,heuristic,display=False)
        exact, lootimes = looMRF.full_data_fit(r_init=full_data_r,get_visuals=False,compute_sensitivity=False)
        exactpredictives[i], t = fullMRF.loo_predictive(missing_site=i, params=exact, display=True)
        exacttime += lootimes['fit'] + t
        results_dict[str(i)][1,:] = exact
        # get L2 errors
        zeroth_errors[i] = np.linalg.norm(full_data_r-exact)
        first_errors[i] = np.linalg.norm(IJ-exact)

    # get estimation errors 
    plt.figure()
    plt.scatter(range(N),np.log(first_errors),label='IJ i.e. 1st order')    
    plt.scatter(range(N),np.log(zeroth_errors),label='full-data estimate i.e. 0th order')    
    plt.xlabel('Leave-one-out fold',fontsize=15)
    plt.ylabel('Log of L2 residual',fontsize=15)
    plt.title('Error of approximate CV %s, beta = %.2f \n IJ runtime = %.2f, LOO runtime = %.2f' %(path,beta,IJtime,exacttime),fontsize=15)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.legend(fontsize=15)
    plt.show()
    
    savepath = "figures/estimation-data-%s-beta-%.2f.png" %(path,beta)
    plt.savefig(savepath)
    
    # get predictive errors
    plt.figure(figsize=(5,5))
    plt.scatter(-exactpredictives, -IJpredictives,color='r')
    cap = max(max(-exactpredictives), max(-IJpredictives))
    ticks = np.linspace(0,cap,40)
    plt.plot(ticks,ticks,linestyle='dashed',color='b')
    plt.title('%s, beta = %.2f' %(path,beta),fontsize=18)
    plt.xlabel('Exact',fontsize=20)
    plt.ylabel('IJ',fontsize=20)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.show()
    
    savepath = "figures/prediction-data-%s-beta-%.2f.png" %(path,beta)
    plt.savefig(savepath)
    
    results_dict['IJtime'] = IJtime
    results_dict['exacttime'] = exacttime
    results_dict['IJpredictives'] = IJpredictives
    results_dict['exactpredictives'] = exactpredictives
    
    savepath = "results/data-%s-beta-%.2f.npz" %(path,beta)
    np.savez(savepath, **results_dict)

    # print runtimes
    ## IJ runtime = full-data fit + sensitivity
    print("Run time of IJ %.2f" %IJtime)

    ## exact refit runtimes = full_data_fit + loo_fits 
    print("Run time of exact LOO %.2f" %exacttime)