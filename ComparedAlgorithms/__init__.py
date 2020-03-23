# -*- coding: utf-8 -*-
""" A solvers' factory method

This module contains the factory method "get_method" which creates the requested solvers for an experiment.
See the following examples on creating solvers.

Example
-------
    get_methods(AlgorithmsType.LinearRegression, []) - Creates all available solvers for Linear-Regression.
    get_methods(AlgorithmsType.LinearRegression, [LinearRegressionMethods.SVDBased]) - Creates only the standard Numpy
     solver for Linear-Regression.

"""

from Infrastructure.utils import List, Dict, is_empty
from Infrastructure.enums import AlgorithmsType, LinearRegressionMethods, LassoRegressionMethods,\
    RidgeRegressionMethods, ElasticNetRegressionMethods
from ComparedAlgorithms import linear_regression, lasso_regression, ridge_regression, elastic_net_regression


# A private dictionary used for get_method.
_algorithms_type_to_algorithms: Dict = {
    AlgorithmsType.LinearRegression: (LinearRegressionMethods, linear_regression),
    AlgorithmsType.RidgeRegression: (RidgeRegressionMethods, ridge_regression),
    AlgorithmsType.LassoRegression: (LassoRegressionMethods, lasso_regression),
    AlgorithmsType.ElasticNetRegression: (ElasticNetRegressionMethods, elastic_net_regression)
}


def get_methods(requested_algorithms_type: AlgorithmsType, compared_methods: List) -> List:
    """
    A factory which creates the requested solvers.

    Attributes:
    -----------
        requested_algorithms_type(AlgorithmsType): The name of the solvers type, i.e linear-regression
        compared_methods(List): A list of specific solvers to create. If empty, all solvers of a given type are created.

    Returns:
    --------
         A list of all relevant solvers.

    """
    solvers_names_list, solvers_factory = _algorithms_type_to_algorithms[requested_algorithms_type]
    solvers: List = []
    if is_empty(compared_methods):
        for solver_name in solvers_names_list:
            solvers.append(solvers_factory.get_method(solver_name))
    else:
        for solver_name in compared_methods:
            solvers.append(solvers_factory.get_method(solver_name))
    return solvers
