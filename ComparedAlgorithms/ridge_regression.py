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
from ComparedAlgorithms.method_boosters import cholesky_booster, caratheodory_booster


@ex.capture
def _sklearn_ridge_regression(data_features: Matrix, output_samples: ColumnVector, n_alphas: int,
                              cross_validation_folds: int, calc_residuals: bool = True) -> (ColumnVector, ColumnVector):
    """
    The standard solver of Scikit-Learn for Ridge-Regression.

    Args:
        data_features(Matrix): The input data matrix ``nxd``.
        output_samples(ColumnVector): The output for the given inputs, ``nx1``.
        n_alphas(int): The number of total regularization terms which will be tested by this solver.
        cross_validation_folds(int): The number of cross-validation folds used in this solver.
        calc_residuals(bool): A flag for calculating the regression residuals. Defaults to ``True``.

    Returns:
        A column vector of the estimated coefficients and the estimator's residuals.

    """
    alphas: ColumnVector = np.random.randn(n_alphas)
    model = RidgeCV(alphas=alphas, cv=cross_validation_folds).fit(data_features, output_samples)
    residuals: ColumnVector = output_samples - model.predict(data_features) if calc_residuals else -1
    return model.coef_, residuals


_sketch_cholesky_ridge_regression: Callable = cholesky_booster(_sklearn_ridge_regression)
_caratheodory_boosted_ridge_regression: Callable = caratheodory_booster(_sklearn_ridge_regression,
                                                                        perform_normalization=False)

# TODO: Append more solvers when they are implemented and tested.
# A private dictionary used for creating the solvers factory :func:`get_method`.
_ridge_regressions_methods: Dict[str, Callable] = {
    RidgeRegressionMethods.SkLearnRidgeRegression: _sklearn_ridge_regression,
    RidgeRegressionMethods.BoostedRidgeRegression: _caratheodory_boosted_ridge_regression,
    RidgeRegressionMethods.SketchAndCholesky: _sketch_cholesky_ridge_regression
}

# A factory which creates the requested Ridge-Regression solvers.
get_method: Callable = create_factory(_ridge_regressions_methods, are_methods=True)
