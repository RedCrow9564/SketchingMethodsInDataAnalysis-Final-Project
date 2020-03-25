# -*- coding: utf-8 -*-
"""
linear_regression.py - Linear-Regression solvers module
=======================================================

This module contains all available Linear-Regression solvers.
These solvers can be received by using the only public method :func:`get_method`.

Example:
    get_method(LinearRegressionMethods.SVDBased) - Creating the standard Numpy solver for Linear-Regression.

"""

from numpy.linalg import lstsq, inv, pinv, qr
import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import lsqr
from Infrastructure.enums import LinearRegressionMethods
from Infrastructure.utils import ex, create_factory, Dict, Scalar, ColumnVector, Matrix, Callable
from ComparedAlgorithms.method_boosters import cholesky_booster, caratheodory_booster, \
    create_coreset_fast_caratheodory, fast_caratheodory_set_python
from ComparedAlgorithms.sketch_preconditioner import generate_sketch_preconditioner


def _svd_based_linear_regression(data_features: Matrix, output_samples: ColumnVector,
                                 calc_residuals: bool = True, **kwargs) -> (ColumnVector, ColumnVector):
    """
    The standard solver of Numpy for Linear-Regression, based on SVD decomposition.

    Args:
        data_features(Matrix): The input data matrix ``nxd``.
        output_samples(ColumnVector): The output for the given inputs, ``nx1``.
        calc_residuals(bool): A flag for calculating the regression residuals. Defaults to ``True``.

    Returns:
        Two column vectors of the estimated coefficients and the estimator's residuals.

    """
    coefficients: ColumnVector = lstsq(data_features, output_samples, rcond=-1)[0]
    residuals: ColumnVector = output_samples - data_features.dot(coefficients) if calc_residuals else -1
    return coefficients, residuals


def _qr_based_linear_regression(data_features: Matrix, output_samples: ColumnVector,
                                calc_residuals: bool = True, **kwargs) -> (ColumnVector, ColumnVector):
    """
    A solver for Linear-Regression, based on QR-decomposition.

    Args:
        data_features(Matrix): The input data matrix ``nxd``.
        output_samples(ColumnVector): The output for the given inputs, ``nx1``.
        calc_residuals(bool): A flag for calculating the regression residuals. Defaults to ``True``.

    Returns:
        Two column vectors of the estimated coefficients and the estimator's residuals.

    """
    q, r = qr(data_features)
    coefficients: ColumnVector = solve_triangular(r, q.T.dot(output_samples), lower=False, check_finite=False)
    residuals: ColumnVector = output_samples - data_features.dot(coefficients) if calc_residuals else -1
    return coefficients, residuals


def _normal_equations_based_linear_regression(data_features: Matrix, output_samples: ColumnVector,
                                              calc_residuals: bool = True, **kwargs) -> (ColumnVector, ColumnVector):
    """
    A solver for Linear-Regression, based on solving the Normal-Equations.

    Args:
        data_features(Matrix): The input data matrix ``nxd``.
        output_samples(ColumnVector): The output for the given inputs, ``nx1``.
        calc_residuals(bool): A flag for calculating the regression residuals. Defaults to ``True``.

    Returns:
        Two column vectors of the estimated coefficients and the estimator's residuals.

    """

    coefficients: ColumnVector = pinv(data_features).dot(output_samples)
    residuals: ColumnVector = output_samples - data_features.dot(coefficients) if calc_residuals else -1
    return coefficients, residuals


@ex.capture(prefix="sketch_preconditioned_config")
def _sketch_preconditioned_based_linear_regression(
        data_features: Matrix, output_samples: ColumnVector, sampled_rows: float, switch_sign_probability: float,
        _seed, calc_residuals: bool = True, **kwargs) -> (ColumnVector, ColumnVector):
    """
    A solver for Linear-Regression, based on the iterative method LSQR and preconditioning.

    Args:
        data_features(Matrix): The input data matrix ``nxd``.
        output_samples(ColumnVector): The output for the given inputs, ``nx1``.
        calc_residuals(bool): A flag for calculating the regression residuals. Defaults to ``True``.

    Returns:
        Two column vectors of the estimated coefficients and the estimator's residuals.

    """

    _, R = qr(generate_sketch_preconditioner(data_features, int(sampled_rows * len(data_features)),
                                             np.empty_like(data_features), _seed, switch_sign_probability))
    partial_solution: ColumnVector = lsqr(data_features.dot(inv(R)), output_samples, atol=1e-15, btol=1e-15)[0]
    coefficients: ColumnVector = solve_triangular(R, partial_solution, lower=False, check_finite=False)
    residuals: ColumnVector = output_samples - data_features.dot(coefficients) if calc_residuals else -1
    return coefficients, residuals


@ex.capture
def _sketch_inverse_linear_regression(data_features: Matrix, output_samples: ColumnVector, clusters_count: int,
                                      calc_residuals: bool = True, **kwargs) -> (ColumnVector, ColumnVector):
    """
    A solver for Linear-Regression, based on boosting the algorithm which solves the Normal-Equations,
    using fast Caratheodory method.

    Args:
        data_features(Matrix): The input data matrix ``nxd``.
        output_samples(ColumnVector): The output for the given inputs, ``nx1``.
        calc_residuals(bool): A flag for calculating the regression residuals. Defaults to ``True``.

    Returns:
        Two column vectors of the estimated coefficients and the estimator's residuals.

    Raises:
        IOError if ``output_samples`` has any negative entries.

    """
    if not np.all(output_samples > 0):
        raise IOError("The estimated values MUST all be non-negative!!")
    coreset: Matrix = create_coreset_fast_caratheodory(data_features, clusters_count)
    outputs_sum: Scalar = np.sum(output_samples)
    points, weights = fast_caratheodory_set_python(data_features.T, output_samples / outputs_sum, clusters_count)
    a_times_outputs: ColumnVector = outputs_sum * points.dot(weights)
    coefficients: ColumnVector = inv(coreset.T.dot(coreset)).dot(a_times_outputs)
    residuals: ColumnVector = output_samples - data_features.dot(coefficients) if calc_residuals else -1
    return coefficients, residuals


_sketch_cholesky_linear_regression: Callable = cholesky_booster(_svd_based_linear_regression)
_caratheodory_booster_linear_regression: Callable = caratheodory_booster(_svd_based_linear_regression,
                                                                         perform_normalization=False)

# A private dictionary used for creating the solvers factory :func:`get_method`.
_linear_regressions_methods: Dict[str, Callable] = {
    LinearRegressionMethods.SVDBased: _svd_based_linear_regression,
    LinearRegressionMethods.QRBased: _qr_based_linear_regression,
    LinearRegressionMethods.NormalEquationsBased: _normal_equations_based_linear_regression,
    LinearRegressionMethods.SketchAndCholesky: _sketch_cholesky_linear_regression,
    LinearRegressionMethods.BoostedSVDSolver: _caratheodory_booster_linear_regression,
    LinearRegressionMethods.SketchAndInverse: _sketch_inverse_linear_regression,
    LinearRegressionMethods.SketchPreconditioned: _sketch_preconditioned_based_linear_regression
}

# A factory which creates the requested linear-regression solvers.
get_method: Callable = create_factory(_linear_regressions_methods, are_methods=True)
