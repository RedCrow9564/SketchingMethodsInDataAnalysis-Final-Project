#!/usr/bin/python
# -*- coding: utf-8 -*-
""" The main module of the project

TODO: Complete.
"""

from Infrastructure.utils import ex, DataLog, List, Dict
from Infrastructure.enums import LogFields, AlgorithmsType, NumpyDistribution, DatabaseType, ExperimentType, \
    LinearRegressionMethods, LassoRegressionMethods, RidgeRegressionMethods, ElasticNetRegressionMethods
from ComparedAlgorithms import get_methods
from database_loader import get_data
from experiments import create_experiment


def _choose_clusters_num(database_type: str, synthetic_data_dim: int) -> int:
    data_dim: int = 1
    if database_type == DatabaseType.Synthetic:
        data_dim = synthetic_data_dim
    elif database_type == DatabaseType.ThreeDRoadNetwork:
        data_dim = 2
    elif database_type == DatabaseType.IndividualHouseholdElectricPowerConsumption:
        data_dim = 8
    return 2 * (data_dim + 1) ** 2 + 2


@ex.config
def config():
    """ Config section

    This function contains all possible configuration for all experiments. Full details on each configuration values
    can be found in :mod:`Infrastructure/enums`.
    """

    compared_algorithms_type: AlgorithmsType = AlgorithmsType.LassoRegression
    compared_methods: List = []  # Leave empty for using all solvers.
    numpy_distribution: NumpyDistribution = NumpyDistribution.CPythonDistribution
    used_database: DatabaseType = DatabaseType.ThreeDRoadNetwork
    experiment_type: ExperimentType = ExperimentType.NumberOfAlphasExperiment
    cross_validation_folds: int = 3
    n_alphas: int = 100

    run_time_experiments_config: Dict[str, range] = {
        "run_time_compared_data_sizes": range(100000, 2400000, 100000)
    }
    number_of_alphas_experiments_config: Dict[str, range] = {
        "alphas_range": range(1, 201, 20)
    }

    synthetic_data_config: Dict[str, int] = {
        "data_size": 2400000,
        "features_num": 7
    }
    resources_path: str = r'Resources'
    clusters_count: int = _choose_clusters_num(used_database, synthetic_data_config["features_num"])
    elastic_net_factor: float = 0.5  # Rho factor in Elastic-Net regularization.


@ex.capture
def run_experiment(compared_algorithms_type: AlgorithmsType, compared_methods: List, used_database: DatabaseType,
                   experiment_type: ExperimentType):
    """
    TODO: Complete.

    :param compared_algorithms_type:
    :param compared_methods:
    :param used_database:
    :param experiment_type:
    :return:
    """
    compared_solvers: List = get_methods(compared_algorithms_type, compared_methods)
    data_matrix, output_samples = get_data(used_database)
    results = create_experiment(experiment_type)(compared_solvers, data_matrix, output_samples)
    return results


@ex.automain
def main() -> None:
    """ TODO: Complete
    :return:
    """
    data_log = run_experiment()
