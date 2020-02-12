from Infrastructure.utils import ex, DataLog
from Infrastructure.enums import LogFields, AlgorithmsType, NumpyDistribution


@ex.config
def config():
    compared_algorithms_type = AlgorithmsType.RidgeRegression
    numpy_distribution = NumpyDistribution.CPythonDistribution
    used_databases = "Synthetic"
    cross_validation_folds = 5
    elastic_net_factor = 0  # Row factor in Elastic-Net regularization.


@ex.capture
def run_experiment(least_square_algorithm) -> DataLog:
    data_log = DataLog(LogFields)  # Initiating the log
    print("Some output")
    data_log.append(LogFields.ChosenLeastSquareAlgorithm, least_square_algorithm)
    return data_log


@ex.automain
def main() -> None:
    # data_log = run_experiment()
    # data_log.save_log()
    # plot_results(data_log)
    pass
