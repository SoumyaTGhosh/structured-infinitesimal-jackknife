"""
Define some helper classes: Factor, VE (variable elimination) and
Elimination Ordering. Also define helper functions like extract_folds.
"""
from tqdm import tqdm
from itertools import product
import autograd
import autograd.numpy as np
from autograd.scipy.special import logsumexp
from collections import defaultdict

from pgmpy.extern import tabulate
import networkx as nx 

# ----------------------------------------------------------------------
# factor class to help with partition computation. Adapted from
# https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/factors/discrete/DiscreteFactor.py
class Factor:
    def __init__(self, variables, cardinality, log_values):
        """
        Inputs:
            variables: list, variable names
            cardinality: list, how many discrete values does each 
                variable take
            log_values: list, log-potential for each combination of variables,
                the latter variables' values changing faster
        """
        log_values = np.array(log_values, dtype=float)
        self.variables = list(variables)
        self.cardinality = np.array(cardinality, dtype=int)
        self.log_values = log_values.reshape(self.cardinality)
        return 
    
    def scope(self):
        return self.variables
    
    def get_cardinality(self, variables):
        return {var: self.cardinality[self.variables.index(var)] for var in variables}
    
    def copy(self):
        """
        Returns a copy of the factor.
        """
        # not creating a new copy of self.values and self.cardinality
        # because __init__ methods does that.
        return Factor(
            self.scope(),
            self.cardinality,
            self.log_values
        )
    
    def product(self, phi1):
        """
        Return product factor between self and phi1. This routine
        is repeated many times to compute the potential over the 
        whole graph.
        """
        phi = self.copy()
        phi1 = phi1.copy()

        # modifying phi to add new variables
        extra_vars = set(phi1.variables) - set(phi.variables)
        if extra_vars:
            slice_ = [slice(None)] * len(phi.variables)
            slice_.extend([np.newaxis] * len(extra_vars))
            phi.log_values = phi.log_values[tuple(slice_)]

            phi.variables.extend(extra_vars)

            new_var_card = phi1.get_cardinality(extra_vars)
            phi.cardinality = np.append(
                phi.cardinality, [new_var_card[var] for var in extra_vars]
            )

        # modifying phi1 to add new variables
        extra_vars = set(phi.variables) - set(phi1.variables)
        if extra_vars:
            slice_ = [slice(None)] * len(phi1.variables)
            slice_.extend([np.newaxis] * len(extra_vars))
            phi1.log_values = phi1.log_values[tuple(slice_)]

            phi1.variables.extend(extra_vars)
            # No need to modify cardinality as we don't need it.

        # rearranging the axes of phi1 to match phi
        for axis in range(phi.log_values.ndim):
            exchange_index = phi1.variables.index(phi.variables[axis])
            phi1.variables[axis], phi1.variables[exchange_index] = (
                phi1.variables[exchange_index],
                phi1.variables[axis],
            )
            phi1.log_values = phi1.log_values.swapaxes(axis, exchange_index)

        phi.log_values = phi.log_values + phi1.log_values
        
        return phi
    
    def many_products(self, factors):
        """
        Take a product between self and many factors, returning a new factor.
        Inputs: 
            factors: list of Factor 
        """
        if len(factors) == 0:
            return self.copy()
        else:
            newphi = self.product(factors[0])
            for i in range(1,len(factors)):
                newphi = newphi.product(factors[i])
            return newphi
    
    def marginalize(self, variables, inplace=False):
        """
        Modifies the factor with marginalized values.

        Parameters
        ----------
        variables: list, array-like
            List of variables over which to marginalize.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Returns
        -------
        DiscreteFactor or None: if inplace=True (default) returns None
                        if inplace=False returns a new `DiscreteFactor` instance.
        """

        phi = self if inplace else self.copy()

        for var in variables:
            if var not in phi.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        var_indexes = [phi.variables.index(var) for var in variables]

        index_to_keep = sorted(set(range(len(self.variables))) - set(var_indexes))
        phi.variables = [phi.variables[index] for index in index_to_keep]
        phi.cardinality = phi.cardinality[index_to_keep]

        phi.log_values = logsumexp(phi.log_values, axis=tuple(var_indexes))

        if not inplace:
            return phi
    
    def __str__(self):
        return self._str(phi_or_p="phi", tablefmt="grid")

    def _str(self, phi_or_p="phi", tablefmt="grid", print_state_names=True):
        """
        Generate the string from `__str__` method. Factors are printed with
        values rather than log-values.
        Parameters
        ----------
        phi_or_p: 'phi' | 'p'
                'phi': When used for Factors.
                  'p': When used for CPDs.
        print_state_names: boolean
                If True, the user defined state names are displayed.
        """
        string_header = list(map(str, self.scope()))
        string_header.append(
            "{phi_or_p}({variables})".format(
                phi_or_p=phi_or_p, variables=",".join(string_header)
            )
        )

        value_index = 0
        factor_table = []
        for prob in product(*[range(card) for card in self.cardinality]):
            prob_list = [
                "{s}_{d}".format(s=list(self.variables)[i], d=prob[i])
                for i in range(len(self.variables))
            ]

            prob_list.append(np.exp(self.log_values.ravel()[value_index]))
            factor_table.append(prob_list)
            value_index += 1

        return tabulate(
            factor_table, headers=string_header, tablefmt=tablefmt, floatfmt=".4f"
        )
    
# ----------------------------------------------------------------------
# variable elimination module. Adapted from
# http://pgmpy.org/_modules/pgmpy/inference/ExactInference.html#VariableElimination
class VE():
    def __init__(self, factors):
        """
        Inputs:
            factors: list of Factor
        Outputs: sets the following instance variables
        """
        # get working factors i.e.m the list of potentials that each variable
        # participates in
        self.factors = defaultdict(list)
        for factor in factors:
            for var in factor.variables:
                self.factors[var].append(factor)
        return
    
    def _get_working_factors(self):
        """
        Make copy of working factors.
        """
        working_factors = {
            node: [factor for factor in self.factors[node]] for node in self.factors
        }
        return working_factors

    def _variable_elimination(
        self,
        variables,
        elimination_order,
        joint,
        show_progress,
    ):
        """
        Implementation of a generalized variable elimination.

        Parameters
        ----------
        variables: list, array-like
            variables that are not to be eliminated.

        elimination_order: list (array-like)
            variables in order of being eliminated 
        """
        
        operation = 'marginalize' 
        
        # Step 1: Deal with the input arguments.
        if isinstance(variables, str):
            raise TypeError("variables must be a list of strings")

        # Step 2: Prepare data structures to run the algorithm.
        eliminated_variables = set()
        # Get working factors
        working_factors = self._get_working_factors()

        # Step 3: Run variable elimination
        if show_progress:
            pbar = tqdm(elimination_order)
        else:
            pbar = elimination_order
        
        max_factor_size = 0
        for var in pbar:
            if show_progress:
                pbar.set_description("Eliminating: {var}".format(var=var))
            # Removing all the factors containing the variables which are
            # eliminated (as all the factors should be considered only once)
            factors = [
                factor
                for factor in working_factors[var]
                if not set(factor.variables).intersection(eliminated_variables)
            ]
            phi = factors[0].many_products(factors[1:])
            phi = getattr(phi, operation)([var], inplace=False)
            max_factor_size = max(max_factor_size, len(phi.scope()))
            """
            if show_progress:
                print("Eliminated variable was %s" %var)
                print("\tResulting factor is")
                print(phi)
            """
            del working_factors[var]
            for variable in phi.variables:
                working_factors[variable].append(phi)
            eliminated_variables.add(var)
            
        if show_progress:
            print("Maximum clique formed %d" %max_factor_size)

        # Step 4: Prepare variables to be returned.
        final_distribution = []
        for node in working_factors:
            factors = working_factors[node]
            for factor in factors:
                if not set(factor.variables).intersection(eliminated_variables):
                    final_distribution.append(factor)

        if joint:
            return final_distribution[0].many_products(final_distribution[1:])
        else:
            query_var_factor = {}
            phi = final_distribution[0].many_products(final_distribution[1:])
            for query_var in variables:
                query_var_factor[query_var] = phi.marginalize(
                    list(set(variables) - set([query_var])), inplace=False
                )
            return query_var_factor

    def query(
        self,
        variables,
        elimination_order,
        joint,
        show_progress,
    ):
        """
        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        elimination_order: list
            order of variable eliminations.

        joint: boolean (default: True)
            If True, returns a Joint Distribution over `variables`.
            If False, returns a dict of distributions over each of the `variables`.
        """
        return self._variable_elimination(
            variables=variables,
            elimination_order=elimination_order,
            joint=joint,
            show_progress=show_progress,
        )
    
    def get_lognorm(self, elimination_order, show_progress):
        """
        Compute log normalizer for the joint distribution.
        Inputs:
            elimination_order:
            show_progress: boolean, whether to print progress of VE
        """
        last = elimination_order[-1]
        phi = self.query(variables=[last], elimination_order=elimination_order[:-1], joint=True, show_progress=show_progress)
        logZ = logsumexp(phi.log_values)
        return logZ
    
# ----------------------------------------------------------------------
# elimination order module. Adapted from 
# http://pgmpy.org/_modules/pgmpy/inference/EliminationOrder.html#BaseEliminationOrder
class BaseEliminationOrder:
    """
    Base class for finding elimination orders.
    """

    def __init__(self, G):
        """
        Init method for the base class of Elimination Orders.
        Parameters
        ----------
        G: NetworkX graph 
        """
        self.G = G.copy()
        return

    def cost(self, node):
        """
        The cost function to compute the cost of elimination of each node.
        This method is just a dummy and returns 0 for all the nodes. Actual cost functions
        are implemented in the classes inheriting BaseEliminationOrder.
        Parameters
        ----------
        node: string, any hashable python object.
            The node whose cost is to be computed.
        """
        return 0

    def get_elimination_order(self, nodes=None, show_progress=True):
        """
        Returns the optimal elimination order based on the cost function.
        The node having the least cost is removed first.
        Parameters
        ----------
        nodes: list, tuple, set (array-like)
            The variables which are to be eliminated.
        """
        nodes = self.G.nodes()

        ordering = []
        if show_progress:
            pbar = tqdm(total=len(nodes))
            pbar.set_description("Finding Elimination Order: ")

        while len(self.G.nodes()) > 0:
            # find minimum score node
            scores = {node: self.cost(node) for node in self.G.nodes()}
            min_score_node = min(scores, key=scores.get)
            # add found node to elimination order
            ordering.append(min_score_node)
            # add edges to node's neighbors
            edge_list = self.fill_in_edges(min_score_node,show_progress)
            self.G.add_edges_from(edge_list)
            # remove node from graph
            self.G.remove_node(min_score_node)
            if show_progress:
                pbar.update(1)
        return ordering

    def fill_in_edges(self, node, show_progress=False):
        """
        Return edges needed to be added to the graph if a node is removed.
        Parameters
        ----------
        node: string (any hashable python object)
            Node to be removed from the graph.
        show_progress: boolean, print clique size formed after removal of 
            vertex
        """
        neighbors = list(self.G.neighbors(node))
        degree = len(neighbors)
        edge_list = []
        if (show_progress):
            print("After removing %s, a clique of size %d forms" %(node, degree))
        if (degree > 1):
            for i in range(degree):
                for j in range(degree-1):
                    if not self.G.has_edge(neighbors[i],neighbors[j]):
                        edge_list.append((neighbors[i],neighbors[j]))
        
        return edge_list

class MinNeighbors(BaseEliminationOrder):
    def cost(self, node):
        """
        The cost of a eliminating a node is the number of neighbors it has in the
        current graph.
        """
        return len(list(self.G.neighbors(node)))

class MinFill(BaseEliminationOrder):
    def cost(self, node):
        """
        The cost of a eliminating a node is the number of edges that need to be added
        (fill in edges) to the graph due to its elimination
        """
        return len(self.fill_in_edges(node))