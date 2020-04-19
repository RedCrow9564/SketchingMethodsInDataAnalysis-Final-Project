# -*- coding: utf-8 -*-
"""
linear_regression.py - Linear-Regression solvers module
=======================================================

This module contains all available Linear-Regression solvers.
These solvers can be received by using the only public method :func:`get_method`.

Example:
::
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
from ComparedAlgorithms.base_least_square_solver import BaseSolver


class _SVDSolver(BaseSolver):
    @ex.capture
    def __init__(self, data_features: Matrix, output_samples: ColumnVector, cross_validation_folds: int,
                 n_alphas: int = -1):
        r"""
        The standard solver of Numpy for Linear-Regression.

        Args:
            data_features(Matrix): The input data matrix :math:`n \times d`.
            output_samples(ColumnVector): The output for the given inputs, :math:`n \times 1`.
            n_alphas(int): The number of total regularization terms which will be tested by this solver.
            cross_validation_folds(int): The number of cross-validation folds used in this solver.

        """
        super(_SVDSolver, self).__init__(data_features, output_samples, n_alphas, cross_validation_folds)
        self._model = None

    def fit(self) -> ColumnVector:
        """
        The method which fits the requested model to the given data.
        """
        self._fitted_coefficients = lstsq(self._data_features, self._output_samples, rcond=-1)[0]
        return self._fitted_coefficients


class _QRSolver(BaseSolver):
    @ex.capture
    def __init__(self, data_features: Matrix, output_samples: ColumnVector, cross_validation_folds: int,
                 n_alphas: int = -1):
        r"""
        A solver for Linear-Regression, based on QR-decomposition.

        Args:
            data_features(Matrix): The input data matrix :math:`n \times d`.
            output_samples(ColumnVector): The output for the given inputs, :math:`n \times 1`.
            n_alphas(int): The number of total regularization terms which will be tested by this solver.
            cross_validation_folds(int): The number of cross-validation folds used in this solver.

        """
        super(_QRSolver, self).__init__(data_features, output_samples, n_alphas, cross_validation_folds)
        self._model = None

    def fit(self) -> ColumnVector:
        """
        The method which fits the requested model to the given data.
        """
        q, r = qr(self._data_features)
        self._fitted_coefficients = solve_triangular(r, q.T.dot(self._output_samples), lower=False, check_finite=False)
        return self._fitted_coefficients


class _NormalEquationsSolver(BaseSolver):
    @ex.capture
    def __init__(self, data_features: Matrix, output_samples: ColumnVector, cross_validation_folds: int,
                 n_alphas: int = -1):
        r"""
        A solver for Linear-Regression, based on solving the Normal-Equations.

        Args:
            data_features(Matrix): The input data matrix :math:`n \times d`.
            output_samples(ColumnVector): The output for the given inputs, :math:`n \times 1`.
            n_alphas(int): The number of total regularization terms which will be tested by this solver.
            cross_validation_folds(int): The number of cross-validation folds used in this solver.

        """
        super(_NormalEquationsSolver, self).__init__(data_features, output_samples, n_alphas, cross_validation_folds)
        self._model = None

    def fit(self) -> ColumnVector:
        """
        The method which fits the requested model to the given data.
        """
        self._fitted_coefficients = pinv(self._data_features).dot(self._output_samples)
        return self._fitted_coefficients


class _SketchPreconditioerSolver(BaseSolver):
    @ex.capture
    def __init__(self, data_features: Matrix, output_samples: ColumnVector, cross_validation_folds: int, _seed,
                 n_alphas: int = -1):
        r"""
        A solver for Linear-Regression, based on solving the Normal-Equations.

        Args:
            data_features(Matrix): The input data matrix :math:`n \times d`.
            output_samples(ColumnVector): The output for the given inputs, :math:`n \times 1`.
            n_alphas(int): The number of total regularization terms which will be tested by this solver.
            cross_validation_folds(int): The number of cross-validation folds used in this solver.

        """
        super(_SketchPreconditioerSolver, self).__init__(data_features, output_samples, n_alphas,
                                                         cross_validation_folds)
        self._model = None
        self._seed = _seed

    @ex.capture(prefix="sketch_preconditioned_config")
    def fit(self, sampled_rows: float, switch_sign_probability: float, min_sampled_rows: float) -> ColumnVector:
        """
        The method which fits the requested model to the given data.
        """
        num_of_rows: int = max(int(sampled_rows * len(self._data_features)), int(min_sampled_rows))
        _, R = qr(generate_sketch_preconditioner(self._data_features, num_of_rows, np.empty_like(self._data_features),
                                                 self._seed, switch_sign_probability))
        partial_solution: ColumnVector = lsqr(self._data_features.dot(inv(R)), self._output_samples,
                                              atol=1e-15, btol=1e-15)[0]
        self._fitted_coefficients = solve_triangular(R, partial_solution, lower=False, check_finite=False)
        return self._fitted_coefficients


class _SketchInverseSolver(BaseSolver):
    @ex.capture
    def __init__(self, data_features: Matrix, output_samples: ColumnVector, cross_validation_folds: int,
                 n_alphas: int = -1):
        r"""
        A solver for Linear-Regression, based on boosting the algorithm which solves the Normal-Equations,
        using fast Caratheodory method.

        Args:
            data_features(Matrix): The input data matrix :math:`n \times d`.
            output_samples(ColumnVector): The output for the given inputs, :math:`n \times 1`.
            n_alphas(int): The number of total regularization terms which will be tested by this solver.
            cross_validation_folds(int): The number of cross-validation folds used in this solver.

        """
        super(_SketchInverseSolver, self).__init__(data_features, output_samples, -1, cross_validation_folds)
        self._model = None

    @ex.capture
    def fit(self, clusters_count) -> ColumnVector:
        """
        The method which fits the requested model to the given data.
        """
        coreset: Matrix = create_coreset_fast_caratheodory(self._data_features, clusters_count)
        adapted_data, adapted_output, outputs_sum = self._preprocess_data()
        weights, chosen_indices = fast_caratheodory_set_python(adapted_data.T, adapted_output, clusters_count)
        a_times_outputs: ColumnVector = outputs_sum * adapted_data[chosen_indices, :].T.dot(weights)
        self._fitted_coefficients = inv(coreset.T.dot(coreset)).dot(a_times_outputs)
        return self._fitted_coefficients

    def _preprocess_data(self):
        data_copy: Matrix = self._data_features.copy()
        output_copy: ColumnVector = self._output_samples.copy()
        negative_indices: ColumnVector = np.argwhere(self._output_samples < 0)
        data_copy[negative_indices, :] *= -1
        output_copy[negative_indices] *= -1
        output_sum = np.sum(output_copy)
        return data_copy, output_copy/output_sum, output_sum


_sketch_cholesky_linear_regression: Callable = cholesky_booster(_SVDSolver)
_caratheodory_booster_linear_regression: Callable = caratheodory_booster(_SVDSolver, perform_normalization=False)

# A private dictionary used for creating the solvers factory :func:`get_method`.
_linear_regressions_methods: Dict[str, Callable] = {
    LinearRegressionMethods.SVDBased: _SVDSolver,
    LinearRegressionMethods.QRBased: _QRSolver,
    LinearRegressionMethods.NormalEquationsBased: _NormalEquationsSolver,
    LinearRegressionMethods.SketchAndCholesky: _sketch_cholesky_linear_regression,
    LinearRegressionMethods.BoostedSVDSolver: _caratheodory_booster_linear_regression,
    LinearRegressionMethods.SketchAndInverse: _SketchInverseSolver,
    LinearRegressionMethods.SketchPreconditioned: _SketchPreconditioerSolver
}

# A factory which creates the requested linear-regression solvers.
get_method: Callable = create_factory(_linear_regressions_methods, are_methods=True)
