#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
main.py - The main module of the project
========================================

This module contains the config for the experiment in the "config" function.
Running this module invokes the :func:`main` function, which then performs the experiment and saves its results
to the configured results folder. Example for running an experiment: ``python main.py``

"""

from Infrastructure.utils import ex, List, Dict, DataLog
from Infrastructure.enums import AlgorithmsType, NumpyDistribution, DatabaseType, ExperimentType, \
    LinearRegressionMethods, LassoRegressionMethods, RidgeRegressionMethods, ElasticNetRegressionMethods
from ComparedAlgorithms import get_methods
from database_loader import get_data
from experiments import create_experiment


def _choose_clusters_num(database_type: str, synthetic_data_dim: int) -> int:
    """
    This method determines the number of clusters for coreset computation, using the number of features in the data.
    If the data has :math:`d` features then the number of used clusters is :math:`2(d+1)^{2}+2`.
    """
    data_dim: int = 1
    if database_type == DatabaseType.Synthetic:
        data_dim = synthetic_data_dim
    elif database_type in [DatabaseType.ThreeDRoadNetwork, DatabaseType.IndividualHouseholdElectricPowerConsumption]:
        data_dim = 2
    elif database_type == DatabaseType.HouseSalesInKingCounty:
        data_dim = 8
    return 2 * (data_dim + 1) ** 2 + 2


@ex.config
def config():
    """ Config section

    This function contains all possible configuration for all experiments. Full details on each configuration values
    can be found in :mod:`enums.py`.
    """

    compared_algorithms_type: AlgorithmsType = AlgorithmsType.LinearRegression
    compared_methods: List = []  # Leave empty for using all solvers.
    numpy_distribution: NumpyDistribution = NumpyDistribution.IntelDistribution
    used_database: DatabaseType = DatabaseType.Synthetic
    experiment_type: ExperimentType = ExperimentType.RunTimeExperiment
    cross_validation_folds: int = 1
    n_alphas: int = 100
    reduction_factor: int = 1

    run_time_experiments_config: Dict[str, range] = {
        "run_time_compared_data_sizes": range(int(5000 / reduction_factor), int(15000 / reduction_factor),
                                              int(5000 / reduction_factor)),
        "calc_transpose_dot_residuals": compared_algorithms_type == AlgorithmsType.LinearRegression
    }
    number_of_alphas_experiments_config: Dict[str, range] = {
        "alphas_range": range(1, 221, 20)
    }

    synthetic_data_config: Dict[str, int] = {
        "data_size": int(15000 / reduction_factor),
        "features_num": 7
    }

    sketch_preconditioned_config: Dict[str, float] = {
        "sampled_rows": 0.005,
        "switch_sign_probability": 0.5,
        "min_sampled_rows": 100.0
    }
    resources_path: str = r'Resources'
    results_path: str = r'Results'
    clusters_count: int = _choose_clusters_num(used_database, synthetic_data_config["features_num"])
    elastic_net_factor: float = 0.5  # Rho factor in Elastic-Net regularization.
    is_positive_definite: bool = True


@ex.automain
def run_experiment(compared_algorithms_type: AlgorithmsType, compared_methods: List, used_database: DatabaseType,
                   experiment_type: ExperimentType, results_path: str) -> Dict[str, DataLog]:
    """ The main function of this project

    This functions performs the desired experiment according to the given configuration.
    The function then saves all the experiment results to a csv file in the results folder (given in the configuration).
    """
    compared_solvers: List = get_methods(compared_algorithms_type, compared_methods)
    data_matrix, output_samples = get_data(used_database)
    experiment = create_experiment(experiment_type)
    all_logs: Dict[str, DataLog] = experiment(compared_solvers, data_matrix, output_samples)

    for log_name, log in all_logs.items():
        log.save_log(log_name, results_path)

    return all_logs
