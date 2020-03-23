# -*- coding: utf-8 -*-
""" ElasticNet-Regression solvers module.

This module contains all available ElasticNet-Regression solvers. These solvers can be received by using the only public
method 'get_method'.

Example:
-------
    get_method(ElasticNetRegressionMethods.SkLearnLassoRegression) - Creating the Scikit-Learn solver
     for ElasticNet-Regression.

"""

from sklearn.linear_model import ElasticNetCV
from Infrastructure.enums import ElasticNetRegressionMethods
from Infrastructure.utils import ex, create_factory, Dict, Scalar, ColumnVector, Matrix, Callable
from ComparedAlgorithms.method_boosters import cholesky_booster, caratheodory_booster


@ex.capture
def _sklearn_elastic_net_regression(data_features: Matrix, output_samples: ColumnVector, n_alphas: int,
                                    cross_validation_folds: int, elastic_net_factor: Scalar, _rnd,
                                    calc_residuals: bool = True) -> (ColumnVector, ColumnVector):
    """
    The standard solver of Scikit-Learn for Elastic-Net-Regression.

    Attributes:
    -----------
        data_features(Matrix): The input data matrix nxd.
        output_samples(ColumnVector): The output for the given inputs, nx1.
        n_alphas(int): The number of total regularization terms which will be tested by this solver.
        cross_validation_folds(int): The number of cross-validation folds used in this solver.
        calc_residuals(bool): A flag for calculating the regression residuals. Defaults to True.

    Returns:
    --------
        A column vector of the estimated coefficients and the 2-norm of the estimator's residuals.

    """
    model = ElasticNetCV(cv=cross_validation_folds, n_alphas=n_alphas, l1_ratio=elastic_net_factor,
                         random_state=_rnd).fit(data_features, output_samples)
    residuals: ColumnVector = output_samples - model.predict(data_features) if calc_residuals else -1
    return model.coef_, residuals


_sketch_cholesky_elastic_net_regression: Callable = cholesky_booster(_sklearn_elastic_net_regression)
_caratheodory_boosted_elastic_net_regression: Callable = caratheodory_booster(_sklearn_elastic_net_regression,
                                                                              perform_normalization=True)

# TODO: Append more solvers when they are implemented and tested.
# A private dictionary used for creating the solvers factory 'get_method'.
_elastic_net_regressions_methods: Dict[str, Callable] = {
    ElasticNetRegressionMethods.SkLearnElasticNetRegression: _sklearn_elastic_net_regression,
    ElasticNetRegressionMethods.BoostedElasticNetRegression: _caratheodory_boosted_elastic_net_regression,
    ElasticNetRegressionMethods.SketchAndCholesky: _sketch_cholesky_elastic_net_regression
}

# A factory which creates the relevant Elastic-Net-Regression solver.
get_method: Callable = create_factory(_elastic_net_regressions_methods, are_methods=True)
