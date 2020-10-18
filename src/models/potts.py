import autograd
import autograd.numpy as np
from autograd.scipy.special import logsumexp
from scipy.special import gammaln
import time
import matplotlib.pyplot as plt
from itertools import product

from pgmpy.extern import tabulate
import random
import networkx as nx

from models.mrf_utils import Factor
from models.mrf_utils import VE
from models.mrf_utils import MinFill

# -----------------------------------------------------------------------
class Potts:
    def __init__(self, data, beta, heuristic, display=False):
        """
        Set up MRF, do some processing to get rough estimate of class means,
        use a heuristic to find a good variable elimination order, report 
        the maximum clique size formed during elimination.

        Inputs:
            data: dictionary with y and W keys
            beta: scalar, connectivity of lattice MRF
            heuristic: str, MinDegree or MinFill
            display: boolean, whether to print information about field
        """
        self._y = data['y']
        self._N = len(self._y)
        self._W = np.asarray(data['W'], dtype=int)
        self._beta = beta
        self._lowmean = np.quantile(self._y, 0.25)
        self._highmean = np.quantile(self._y, 0.5)
        if (display):
            print("There are %d sites" %len(self._y)) 
            print("Dimensions of adjacency matrix (%d, %d)" %(self._W.shape[0],self._W.shape[1])) 
            list_of_counts = ["(site %d, y %d)" %(i,self._y[i]) for i in range(self._N)]
            print("Counts")
            print(list_of_counts)
            print("Adjacency matrix")
            print(self._W)
            print("Initial guess of class means %.2f and %.2f" %(self._lowmean, self._highmean))
        self.params_ones = None
        self.dParams_dWeights = None

        # use heuristic to find a good elimination order
        if (heuristic == "MinDegree"):
            degrees = np.sum(self._W,axis=1)
            nodes = range(self._N)
            temp = list(zip(nodes, degrees))
            temp = sorted(temp, key=lambda tup: tup[1])
            elimination_order = ["x%d" %d[0] for d in temp]
        else:
            G = nx.from_numpy_matrix(self._W)
            mapping = {i: "x%d" %i for i in range(len(G.nodes()))}
            G = nx.relabel_nodes(G,mapping)
            nodes = G.nodes()
            if (heuristic == "MinFill"):
                elimination_order = MinFill(G).get_elimination_order(nodes, display)
            elif (heuristic == "MinNeighbors"):
                elimination_order = MinNeighbors(G).get_elimination_order(nodes, display)
        self._elimination_order = elimination_order
        
        if (display):
            print("Variable elimination will use the following order")
            print(elimination_order)
        
        # check size of biggest clique encountered during elimination
        factors = []
        # unary potentials (in case some nodes are disconnected from other nodes)
        for i in range(self._N):
            node0 = "x%d" %(i)
            factor = Factor([node0],cardinality=[2],log_values=[0,0])
            factors.append(factor)
            
        # pairwise potentials
        for i in range(self._N):
            for j in range(self._N):
                if self._W[i,j] == 1:
                    node0 = "x%d" %(i)
                    node1 = "x%d" %(j)
                    factor = Factor([node0,node1],cardinality=[2,2],log_values=[self._beta,0,0,self._beta])
                    factors.append(factor)
                    
        inference = VE(factors)
        t0 = time.time()
        self._logZ = inference.get_lognorm(elimination_order,display)
        t1 = time.time()
        print("Time of one normalization constant computation %.2f" %(t1-t0))
        print("Finished initialization\n")
        return

    def get_sensitivity(self):
        """
        Return sensitivitiy around full-data solution.
        """
        assert self.dParams_dWeights is not None
        return self.dParams_dWeights

    def _get_elimination_order(self):
        """
        Return copy of elimination order.
        """
        return self._elimination_order.copy()
    
    def get_neighbors(self, i):
        list_of_neighbors = ["(site %d, y %d)" %(j,self._y[j]) for j in range(self._N) if self._W[i,j] == 1]
        return list_of_neighbors

    def full_data_fit(self, r_init, get_visuals, compute_sensitivity, max_iter=100, var_converge=1e-8):
        """
        Estimate class means using full data and compute relevant sensitivities for 
        IJ approximation. Report class means and relevant runtimes.
        """
        t0 = time.time()
        if (r_init is None):
            r_init = np.array([self._lowmean, self._highmean])
        self.params_ones = self.EM(r_init, get_visuals, max_iter, var_converge)
        t1 = time.time()
        times = {}
        times['fit'] = t1-t0
        print("Time to fit %.2f" %times['fit'])
        if (compute_sensitivity):
            t0 = time.time()
            self.dParams_dWeights = self.compute_dParams_dWeights(np.ones(self._N),self.params_ones)
            t1 = time.time()
            times['sensitivity'] = t1-t0
            print("Time to compute sensitivity %.2f" %times['sensitivity'])
        print("Finished fitting\n")
        return self.params_ones, times

    # -----------------------------------------------------------------------------------------
    # EM code
    def EM(self, r_init, get_visuals, max_iter, var_converge):
        """
        EM to maximize log(y; r0, r1) w.r.t. r0 and r1
        Input:
            r_init: list, class_means
        """
        # initialize using lowmean and highmean
        params = r_init
        LLlog = []
        iteration = 0
        prevLL = self.LL(params)
        LLlog.append(prevLL)
        while True:
            iteration += 1
            # E-step
            q = self.Estep(params)
            # M-step
            r = self.Mstep(q)
            params = r
            LL = self.LL(params)
            LLlog.append(LL)
            if (iteration > max_iter or abs((LL - prevLL)/prevLL) < var_converge):
                break
            prevLL = LL
        if (get_visuals):
            # plot log LL of data as function of EM iteration
            plt.figure()
            plt.plot(range(len(LLlog)),LLlog, marker='o')
            plt.xlabel("Iteration",fontsize=15)
            plt.ylabel("Log-likelihood",fontsize=15)
            plt.tick_params(axis='x', labelsize=15)
            plt.tick_params(axis='y', labelsize=15)
            plt.show()
        # report final class means
        print("Inital means of EM r0 = %.2f, r1 = %.2f" %(r_init[0],r_init[1]))
        print("\tFinal means of EM r0 = %.2f, r1 = %.2f" %(r[0],r[1]))
        return r

    def Estep(self, params, printfactors=False):
        """
        Return the marginal distributions p(x_i|y;params). Because binary variable,
        only report p(x_i=0|y;params).
        """
        r0 = params[0]
        r1 = params[1]
                
        # create list of factors representing p(x|y;params)
        factors = []
        
        # unary potentials in the prior (in case some nodes are disconnected from other nodes)
        for i in range(self._N):
            node0 = "x%d" %(i)
            factor = Factor([node0],cardinality=[2],log_values=[0,0])
            factors.append(factor)
        
        ## pairwise potentials in the prior 
        for i in range(self._N):
            for j in range(self._N):
                if self._W[i,j] == 1:
                    node0 = "x%d" %(i)
                    node1 = "x%d" %(j)
                    prior_factor = Factor([node0,node1],cardinality=[2,2],log_values=[self._beta,0,0,self._beta])
                    factors.append(prior_factor)
                        
        ## add observations to make the joint model
        for i in range(self._N):
            node = "x%d" %(i)
            low_class = -r0 + self._y[i]*np.log(r0)-gammaln(self._y[i]+1)
            high_class = -r1 + self._y[i]*np.log(r1)-gammaln(self._y[i]+1)
            observation_factor = Factor([node],cardinality=[2],log_values=[low_class, high_class])
            factors.append(observation_factor)

        inference = VE(factors)
        
        # marginals p(x_i|y;params)
        q = np.zeros(self._N) # q(x_i^0)
        for i in range(self._N):
            node = "x%d" %i
            elimination_order = self._get_elimination_order()
            elimination_order.remove(node)
            marginal_factor = inference.query(variables=[node],elimination_order=elimination_order,joint=True,show_progress=False)
            log_norm = logsumexp(marginal_factor.log_values)
            # properly normalize the marginal factors
            marginal_factor.log_values = marginal_factor.log_values - log_norm
            if (printfactors):
                print(marginal_factor)
            q[i] = np.exp(marginal_factor.log_values[0])
        return q

    def Mstep(self, low):
        """
        Update class means.
        Inputs: 
            low: list, marginal distributions q(x_i = 0|y;params) 
        """
        high = 1-low
        r0 = np.sum(low*self._y)/(np.sum(low)+1e-100)
        r1 = np.sum(high*self._y)/(np.sum(high)+1e-100)
        r = np.array([r0, r1])
        return r

    # regular log likelihood 
    def LL(self, params):
        """
        Compute log p(y;params) = log sum_x p(y,x;params) 
        where 
            - p(y|x) is product of independent Poisson with two classes of means r0, r1 
            - p(x) is Potts model with uniform connection strength beta: 
            p(x) propto exp(beta sum_{i,j} W_{ij} {x_i = x_j})

        Inputs:
            params: list - r0, r1
        """
        r0 = params[0]
        r1 = params[1]
                
        # create list of factors representing p(x|y;params)
        factors = []
        
         # unary potentials in the prior (in case some nodes are disconnected from other nodes)
        for i in range(self._N):
            node0 = "x%d" %(i)
            factor = Factor([node0],cardinality=[2],log_values=[0,0])
            factors.append(factor)
        
        ## iterate through all edges to create the prior model
        for i in range(self._N):
            for j in range(self._N):
                if self._W[i,j] == 1:
                    node0 = "x%d" %(i)
                    node1 = "x%d" %(j)
                    prior_factor = Factor([node0,node1],cardinality=[2,2],log_values=[self._beta,0,0,self._beta])
                    factors.append(prior_factor)
                        
        ## add observations to make the joint model
        for i in range(self._N):
            node = "x%d" %(i)
            low_class = -r0 + self._y[i]*np.log(r0)-gammaln(self._y[i]+1)
            high_class = -r1 + self._y[i]*np.log(r1)-gammaln(self._y[i]+1)
            observation_factor = Factor([node],cardinality=[2],log_values=[low_class, high_class])
            factors.append(observation_factor)

        inference = VE(factors)
        elimination_order = self._get_elimination_order()
        log_numerator = inference.get_lognorm(elimination_order,show_progress=False)
        return log_numerator-self._logZ

    # -------------------------------------------------------------------------------------------
    # IJ code
    def weightedLL(self, params, weights):
        """
        Compute log p(y;weights,params) = log sum_x p(y,x;weights,params) 
        where 
            - p(y|x; weights) is product of independent Poisson with two classes of means r0, r1 
            over present observations i.e. weights[i] = 1.
            - p(x) is Potts model with uniform connection strength beta: 
            p(x) propto exp(beta sum_{i,j} weights[i] * weights[j] W_{ij} {x_i = x_j})

        Inputs:
            params: list - r0, r1
            weights: list, whether sites are present in model
        """
        r0 = params[0]
        r1 = params[1]
        elimination_order = self._get_elimination_order()

        denom_factors = []
        # iterate through all edges to create the prior model
        for i in range(self._N):
            for j in range(self._N):
                if self._W[i,j] == 1:
                    node0 = "x%d" %(i)
                    node1 = "x%d" %(j)
                    inclusion_weight = weights[i]*weights[j]
                    log_values = inclusion_weight*np.array([self._beta,0,0,self._beta])
                    prior_factor = Factor([node0,node1],cardinality=[2,2],log_values=log_values)
                    denom_factors.append(prior_factor)

        prior_inference = VE(denom_factors)
        log_denominator = prior_inference.get_lognorm(elimination_order,show_progress=False)
                        
        # add the observations to make the joint model
        numer_factors = denom_factors.copy()
        for i in range(self._N):
            node = "x%d" %(i)
            low_class = -r0 + self._y[i]*np.log(r0)-gammaln(self._y[i]+1)
            high_class = -r1 + self._y[i]*np.log(r1)-gammaln(self._y[i]+1)
            log_values = weights[i]*np.array([low_class, high_class])
            observation_factor = Factor([node],cardinality=[2],log_values=log_values)
            numer_factors.append(observation_factor)

        joint_inference = VE(numer_factors)
        log_numerator = joint_inference.get_lognorm(elimination_order,show_progress=False)
        return log_numerator-log_denominator

    # compute hessian of weighted log-likelihood w.r.t parameters
    def compute_hessian(self, weights, params):
        eval_hess = autograd.hessian(self.weightedLL, argnum=0)
        hess = eval_hess(params, weights)
        return hess
    
    # sensitivity around weights
    def compute_dParams_dWeights(self, weights, params):
        hess = self.compute_hessian(weights, params)
        eval_d2l_dParams_dWeights = \
                autograd.jacobian(autograd.jacobian(self.weightedLL, argnum=0),
                                  argnum=1)
        d2l_dParams_dWeights = eval_d2l_dParams_dWeights(params, weights)
        sensitivity = -np.linalg.solve(hess, d2l_dParams_dWeights)
        return sensitivity
    
    def retrain_with_weights(self, weights):
        assert self.dParams_dWeights is not None
        assert self.params_ones is not None
        t0 = time.time()
        IJ = (self.params_ones +
                  self.dParams_dWeights.dot(weights-1))
        t1 = time.time()
        return IJ, t1-t0

    # -------------------------------------------------------------------------------------------------
    # predictive code
    def loo_predictive(self, missing_site, params, display=False):
        """
        Compute log p(y_i|y_{-i};params).
        Inputs:
            missing_site: scalar, location of missing site
            params: list, class means
            display: boolean, whether to plot factors 
        Outputs:
            log likelihood and how much time it took to compute
        """
        t0 = time.time()
        # compute log p(x|y_{-i};params)
        r0 = params[0]
        r1 = params[1]
        
        numer_factors = []
        
         # unary potentials in the prior (in case some nodes are disconnected from other nodes)
        for i in range(self._N):
            node0 = "x%d" %(i)
            factor = Factor([node0],cardinality=[2],log_values=[0,0])
            numer_factors.append(factor)
        
        ## iterate through all edges to create the prior model
        for i in range(self._N):
            for j in range(self._N):
                if self._W[i,j] == 1:
                    node0 = "x%d" %(i)
                    node1 = "x%d" %(j)
                    prior_factor = Factor([node0,node1],cardinality=[2,2],log_values=[self._beta,0,0,self._beta])
                    numer_factors.append(prior_factor)
                        
        # add non-missing observations to make the joint model
        for i in range(self._N):
            if (i != missing_site):
                node = "x%d" %(i)
                low_class = -r0 + self._y[i]*np.log(r0)-gammaln(self._y[i]+1)
                high_class = -r1 + self._y[i]*np.log(r1)-gammaln(self._y[i]+1)
                observation_factor = Factor([node],cardinality=[2],log_values=[low_class, high_class])
                numer_factors.append(observation_factor)
            
        inference = VE(numer_factors)
        # compute log p(x_i|y_{-i};params) by marginalizing x_{-i} in log p(x|y_{-i};params)
        node = "x%d" %missing_site
        elimination_order = self._get_elimination_order()
        elimination_order.remove(node)
        marginal_factor = inference.query(variables=[node],elimination_order=elimination_order,joint=True,show_progress=False)
        log_norm = logsumexp(marginal_factor.log_values)
        marginal_factor.log_values = marginal_factor.log_values-log_norm
        
        if (display):
            print("Site %d missing. p(x_i|y_{-i}) is" %missing_site)
            print(marginal_factor)
            
        # compute log p(y_i|y_{-i};params)
        low_class = -r0 + self._y[missing_site]*np.log(r0)-gammaln(self._y[missing_site]+1)
        high_class = -r1 + self._y[missing_site]*np.log(r1)-gammaln(self._y[missing_site]+1)
        loglowterm = marginal_factor.log_values[0] + low_class
        loghighterm = marginal_factor.log_values[1] + high_class
        ans = logsumexp([loglowterm, loghighterm])
        t1 = time.time()
        return ans, t1-t0