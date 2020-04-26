from Infrastructure.utils import ColumnVector, Matrix


class BaseSolver(object):
    def __init__(self, data_features: Matrix, output_samples: ColumnVector, n_alphas: int,
                 cross_validation_folds: int):
        self._data_features = data_features
        self._output_samples = output_samples
        self._cross_validation_folds = cross_validation_folds
        self._n_alphas = n_alphas
        self._model = None
        self._fitted_coefficients: ColumnVector = None

    def fit(self) -> ColumnVector:
        raise NotImplementedError("Any subclass MUST implement this method!")

    def calc_residuals(self) -> ColumnVector:
        """
        A method for calculating the estimation errors of the fitted model.
        It can NOT be invoked before the 'fit' method.

        Returns:
            A column vector of the estimated coefficients and the estimator's residuals.

        """
        return self._output_samples - self._data_features.dot(self._fitted_coefficients)
