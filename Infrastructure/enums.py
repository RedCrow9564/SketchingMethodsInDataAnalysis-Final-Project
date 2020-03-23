# -*- coding: utf-8 -*-
""" All enums section

This module contains all possible enums of this project. Most of them are used by the configuration section in main.py.
See the following example on using an enum.

Example
-------
    a = DatabaseType.Synthetic

"""

from typing import Iterator, List
import inspect


class _MetaEnum(type):
    """
    A private meta-class which given any BaseEnum object to be an iterable.
    This can be used for iterating all possible values of this enum. Should not be used explicitly.
    """
    def __iter__(self) -> Iterator:
        """
        This method gives any BaseEnum the ability of iterating over all the enum's values.

        Returns:
        --------
            An iterator for the collection of all the enum's values.

        """
        # noinspection PyUnresolvedReferences
        return self.enum_iter()

    def __contains__(self, item) -> bool:
        """
        This method give any BaseEnum the ability to test if a given item is a possible value for this enum class.

        Returns:
        --------
            A flag which indicates if 'item' is a possible value for this enum class.

        """
        # noinspection PyUnresolvedReferences
        return self.enum_contains(item)


class BaseEnum(metaclass=_MetaEnum):
    """
    A basic interface for all enum classes. Should be sub-classed in eny enum.

    Example:
    -------
        class AlgorithmsType(BaseEnum)

    """

    @classmethod
    def enum_iter(cls) -> Iterator:
        """
        This method gives any BaseEnum the ability of iterating over all the enum's values.

        Returns:
        --------
            An iterator for the collection of all the enum's values.

        """
        return iter(cls.get_all_values())

    @classmethod
    def enum_contains(cls, item) -> bool:
        """
        This method give any BaseEnum the ability to test if a given item is a possible value for this enum class.

        Returns:
        --------
                A flag which indicates if 'item' is a possible value for this enum class.

        """
        return item in cls.get_all_values()

    @classmethod
    def get_all_values(cls) -> List:
        """
        A method which fetches all possible values of an enum. Used for iterating over an enum.

        Returns:
        --------
            A list of all possible enum's values.

        """
        all_attributes: List = inspect.getmembers(cls, lambda a: not inspect.ismethod(a))
        all_attributes = [value for name, value in all_attributes if not (name.startswith('__') or name.endswith('__'))]
        return all_attributes


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
    #LGMRES: str = "LGMRES"  # Scipy's function supports preconditioning.
    #QMR: str = "QMR"  # Scipy's function supports preconditioning.
    #LSQR: str = "LSQR"  # Scipy's function DOESN'T support preconditioning!!!
    #LSMR: str = "LSMR"  # Scipy's function DOESN'T support preconditioning!!!


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
