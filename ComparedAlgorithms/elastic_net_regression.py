# -*- coding: utf-8 -*-
"""
elastic_net_regressions.py - ElasticNet-Regression solvers module
=================================================================

This module contains all available ElasticNet-Regression solvers.
These solvers can be received by using the only public method :func:`get_method`.

Example:
    get_method(ElasticNetRegressionMethods.SkLearnLassoRegression) - Creating the Scikit-Learn solver
     for ElasticNet-Regression.

"""

from sklearn.linear_model import ElasticNetCV
from Infrastructure.enums import ElasticNetRegressionMethods
from Infrastructure.utils import ex, create_factory, Dict, Scalar, ColumnVector, Matrix, Callable
from ComparedAlgorithms.method_boosters import caratheodory_booster
from ComparedAlgorithms.base_least_square_solver import BaseSolver


class _SkLearnElasticNetSolver(BaseSolver):
    @ex.capture
    def __init__(self, data_features: Matrix, output_samples: ColumnVector, n_alphas: int,
                 cross_validation_folds: int, elastic_net_factor: Scalar, _rnd):
        """
        The standard solver of Scikit-Learn for Lasso-Regression.

        Args:
            data_features(Matrix): The input data matrix ``nxd``.
            output_samples(ColumnVector): The output for the given inputs, ``nx1``.
            n_alphas(int): The number of total regularization terms which will be tested by this solver.
            cross_validation_folds(int): The number of cross-validation folds used in this solver.

        """
        super(_SkLearnElasticNetSolver, self).__init__(data_features, output_samples, n_alphas, cross_validation_folds)
        self._model = ElasticNetCV(cv=cross_validation_folds, n_alphas=n_alphas, random_state=_rnd,
                                   l1_ratio=elastic_net_factor, normalize=False)

    def fit(self) -> ColumnVector:
        """
        The method which fits the requested model to the given data.
        """
        self._model.fit(self._data_features, self._output_samples)
        self._fitted_coefficients = self._model.coef_
        return self._fitted_coefficients


_caratheodory_boosted_elastic_net_regression: Callable = caratheodory_booster(_SkLearnElasticNetSolver,
                                                                              perform_normalization=True)

# A private dictionary used for creating the solvers factory 'get_method'.
_elastic_net_regressions_methods: Dict[str, Callable] = {
    ElasticNetRegressionMethods.SkLearnElasticNetRegression: _SkLearnElasticNetSolver,
    ElasticNetRegressionMethods.BoostedElasticNetRegression: _caratheodory_boosted_elastic_net_regression
}

# A factory which creates the relevant Elastic-Net-Regression solver.
get_method: Callable = create_factory(_elastic_net_regressions_methods, are_methods=True)
