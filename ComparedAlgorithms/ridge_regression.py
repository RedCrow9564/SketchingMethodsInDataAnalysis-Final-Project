# -*- coding: utf-8 -*-
"""
ridge_regression.py - Ridge-Regression solvers module
=====================================================

This module contains all available Ridge-Regression solvers.
These solvers can be received by using the only public method :func:`get_method`.

Example:
    get_method(RidgeRegressionMethods.SkLearnLassoRegression) - Creating the Scikit-Learn solver for Ridge-Regression.

"""

from sklearn.linear_model import RidgeCV
import numpy as np
from Infrastructure.enums import RidgeRegressionMethods
from Infrastructure.utils import ex, create_factory, Dict, ColumnVector, Matrix, Callable
from ComparedAlgorithms.method_boosters import caratheodory_booster
from ComparedAlgorithms.base_least_square_solver import BaseSolver


class _SkLearnRidgeSolver(BaseSolver):
    @ex.capture
    def __init__(self, data_features: Matrix, output_samples: ColumnVector, n_alphas: int,
                 cross_validation_folds: int):
        """
        The standard solver of Scikit-Learn for Lasso-Regression.

        Args:
            data_features(Matrix): The input data matrix ``nxd``.
            output_samples(ColumnVector): The output for the given inputs, ``nx1``.
            n_alphas(int): The number of total regularization terms which will be tested by this solver.
            cross_validation_folds(int): The number of cross-validation folds used in this solver.

        """
        super(_SkLearnRidgeSolver, self).__init__(data_features, output_samples, n_alphas, cross_validation_folds)
        alphas: ColumnVector = np.random.randn(n_alphas)
        self._model = RidgeCV(cv=cross_validation_folds, alphas=alphas)

    def fit(self) -> ColumnVector:
        """
        The method which fits the requested model to the given data.
        """
        self._model.fit(self._data_features, self._output_samples)
        self._fitted_coefficients = self._model.coef_
        return self._fitted_coefficients


_caratheodory_boosted_ridge_regression: Callable = caratheodory_booster(_SkLearnRidgeSolver,
                                                                        perform_normalization=False)

# A private dictionary used for creating the solvers factory :func:`get_method`.
_ridge_regressions_methods: Dict[str, Callable] = {
    RidgeRegressionMethods.SkLearnRidgeRegression: _SkLearnRidgeSolver,
    RidgeRegressionMethods.BoostedRidgeRegression: _caratheodory_boosted_ridge_regression
}

# A factory which creates the requested Ridge-Regression solvers.
get_method: Callable = create_factory(_ridge_regressions_methods, are_methods=True)
