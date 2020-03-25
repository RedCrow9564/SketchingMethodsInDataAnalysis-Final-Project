# -*- coding: utf-8 -*-
"""
enums.py - All enums section
============================

This module contains all possible enums of this project. Most of them are used by the configuration section in
:mod:`main`. An example for using enum: ``ExperimentType.RunTimeExperiment``

"""

from Infrastructure.utils import BaseEnum


class LogFields(BaseEnum):
    """
    The enum class of fields within experiments logs.
    """
    DataSize: str = "Data size"
    Coefficients: str = "Coefficients"
    Residuals: str = "Residuals" #  2-norm of errors of estimation for the given data.
    DurationInSeconds: str = "Duration in seconds"
    AtTimesErrors: str = "A transpose times Errors"
    AlphasCount: str = "Alphas count"


class DatabaseType(BaseEnum):
    """
    The enum class of dataset type to use in an experiment.
    """
    Synthetic: str = "Synthetic"  # A random data

    # 3D road network with highly accurate elevation information (+-20cm) from Denmark,
    # used in eco-routing and fuel/Co2-estimation routing algorithms.
    # See https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+(North+Jutland,+Denmark)
    ThreeDRoadNetwork: str = "3D Road Network"

    # It includes homes sold between May 2014 and May 2015. See https://www.kaggle.com/harlfoxem/housesalesprediction
    HouseSalesInKingCounty: str = "House Sales in King County, USA"

    # Measurements of electric power consumption in one household with a one-minute sampling rate over a period of
    # almost 4 years. This archive contains 2075259 measurements gathered in a house located in Sceaux
    # #(7km of Paris, France) between December 2006 and November 2010 (47 months).
    # See https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption.
    IndividualHouseholdElectricPowerConsumption: str = "Individual household electric power consumption"


class ExperimentType(BaseEnum):
    """
    The enum class of experiment types.
    """
    RunTimeExperiment: str = "Run-Time Experiment"
    AccuracyExperiment: str = "Accuracy Experiment"
    NumberOfAlphasExperiment: str = "Number of Alphas Experiment"


class AlgorithmsType(BaseEnum):
    """
    The enum class of algorithms type to be compared in an experiment.
    """
    LinearRegression: str = "Linear Regression"
    LassoRegression: str = "Lasso Regression"
    RidgeRegression: str = "Ridge Regression"
    ElasticNetRegression: str = "ElasticNet Regression"


class LinearRegressionMethods(BaseEnum):
    """
    The enum class of possible Linear-Regression solvers for experiments.
    """
    BoostedSVDSolver: str = "SVD solver with Caratheodory Coreset Booster"
    SVDBased: str = "Based on SVD"  # Numpy's standard linear-regression solver- lstsq, based on SVD decomposition.
    QRBased: str = "Based on QR"  # Solver based on QR decomposition.
    NormalEquationsBased: str = "Based on Normal-Equations"  # Solver based on solving the Normal-Equations.
    SketchAndCholesky: str = "Sketch + Cholesky"
    SketchAndInverse: str = "Sketch + Inverse"
    SketchPreconditioned: str = "Sketch Preconsitioning for LSQR"


class LassoRegressionMethods(BaseEnum):
    """
    The enum class of possible Lasso-Regression solvers for experiments.
    """
    SkLearnLassoRegression: str = "Scikit-Learn's LassoCV Method"  # Scikit-learn standard Lasso-Regression solver.

    # The solver of Scikit-Learn, boosted by Caratheodory coreset booster.
    BoostedLassoRegression: str = "LassoCV with Fast Caratheodory booster"
    SketchAndCholesky: str = "Sketch + Cholesky"  # The solver of Scikit-Learn, boosted by Cholesky decomposition.


class RidgeRegressionMethods(BaseEnum):
    """
    The enum class of possible Ridge-Regression solvers for experiments.
    """
    SkLearnRidgeRegression: str = "Scikit-Learn's RidgeCV Method"  # Scikit-learn standard Ridge-Regression solver.

    # The solver of Scikit-Learn, boosted by Caratheodory coreset booster.
    BoostedRidgeRegression: str = "RidgeCV with Fast Caratheodory booster"
    SketchAndCholesky: str = "Sketch + Cholesky"  # The solver of Scikit-Learn, boosted by Cholesky decomposition.


class ElasticNetRegressionMethods(BaseEnum):
    """
    The enum class of possible Elastic-Net-Regression solvers for experiments.
    """
    # Scikit-learn standard Elastic-Net-Regression solver.
    SkLearnElasticNetRegression: str = "Scikit-Learn's ElasticNetCV Method"

    # The solver of Scikit-Learn, boosted by Caratheodory coreset booster.
    BoostedElasticNetRegression: str = "ElasticNetCV with Fast Caratheodory booster"
    SketchAndCholesky: str = "Sketch + Cholesky"  # The solver of Scikit-Learn, boosted by Cholesky decomposition.


class NumpyDistribution(BaseEnum):
    """
    The type of Numpy distribution used within an experiment. Used for experiment documentation only.
    """
    IntelDistribution: str = "Intel's Distribution"
    CPythonDistribution: str = "CPython's distribution"
