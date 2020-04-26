# -*- coding: utf-8 -*-
"""
lasso_regression.py - Lasso-Regression solvers module
=====================================================

This module contains all available Lasso-Regression solvers.
These solvers can be received by using the only public method :func:`get_method`.

Example:
    get_method(LassoRegressionMethods.SkLearnLassoRegression) - Creating the Scikit-Learn solver for Lasso-Regression.

"""

from sklearn.linear_model import LassoCV
from Infrastructure.enums import LassoRegressionMethods
from Infrastructure.utils import ex, create_factory, Dict, ColumnVector, Matrix, Callable
from ComparedAlgorithms.method_boosters import caratheodory_booster
from ComparedAlgorithms.base_least_square_solver import BaseSolver


class _SkLearnLassoSolver(BaseSolver):
    @ex.capture
    def __init__(self, data_features: Matrix, output_samples: ColumnVector, n_alphas: int,
                 cross_validation_folds: int, _rnd):
        """
        The standard solver of Scikit-Learn for Lasso-Regression.

        Args:
            data_features(Matrix): The input data matrix ``nxd``.
            output_samples(ColumnVector): The output for the given inputs, ``nx1``.
            n_alphas(int): The number of total regularization terms which will be tested by this solver.
            cross_validation_folds(int): The number of cross-validation folds used in this solver.

        """
        super(_SkLearnLassoSolver, self).__init__(data_features, output_samples, n_alphas, cross_validation_folds)
        self._model = LassoCV(cv=cross_validation_folds, n_alphas=n_alphas, random_state=_rnd, normalize=False)

    def fit(self):
        """
        The method which fits the requested model to the given data.
        """
        self._model.fit(self._data_features, self._output_samples)
        self._fitted_coefficients = self._model.coef_
        return self._fitted_coefficients


_caratheodory_boosted_lasso_regression: Callable = caratheodory_booster(_SkLearnLassoSolver, perform_normalization=True)

# A private dictionary used for creating the solvers factory :func:`get_method`.
_lasso_regressions_methods: Dict[str, Callable] = {
    LassoRegressionMethods.SkLearnLassoRegression: _SkLearnLassoSolver,
    LassoRegressionMethods.BoostedLassoRegression: _caratheodory_boosted_lasso_regression
}

# A factory which creates the requested Lasso-Regression solvers.
get_method: Callable = create_factory(_lasso_regressions_methods, are_methods=True)
