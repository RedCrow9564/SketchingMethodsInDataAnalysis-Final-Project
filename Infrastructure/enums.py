from Infrastructure.utils import BaseEnum


class LogFields(BaseEnum):
    ChosenLeastSquareAlgorithm = "Chosen least square algorithm"


class DatabaseType(BaseEnum):
    Synthetic = "Synthetic"  # A random data
    HouseSalesInKingCounty = "House Sales in King County, USA" #  It includes homes sold between May 2014 and May 2015.

    # Measurements of electric power consumption in one household with a one-minute sampling rate over a period of
    # almost 4 years. This archive contains 2075259 measurements gathered in a house located in Sceaux
    # #(7km of Paris, France) between December 2006 and November 2010 (47 months).
    IndividualHouseholdElectricPowerConsumption = "Individual household electric power consumption"


class AlgorithmsType(BaseEnum):
    LinearRegression = "Linear Regression"
    LassoRegression = "Lasso Regression"
    RidgeRegression = "Ridge Regression"
    ElasticNetRegression = "ElasticNet Regression"


class LinearRegressionAlgorithms(BaseEnum):
    BoostedSVDSolver = "SVD solver with Caratheodory Coreset Booster"
    SVDBased = "Based on SVD"  # numpy.linalg.lstaq(rcond=-1)
    QRBased = "Based on QR"  # statsmodels.regression.linear_model.OLS.fit(method='qr')
    LGMRES = "LGMRES"  # Scipy's function supports preconditioning.
    QMR = "QMR"  # Scipy's function supports preconditioning.
    LSQR = "LSQR"  # Scipy's function DOESN'T support preconditioning!!!
    LSMR = "LSMR"  # Scipy's function DOESN'T support preconditioning!!!


class LassoRegressionMethods(BaseEnum):
    SkLearnLassoRegression = "Scikit-Learn's LassoCV Method"
    BoostedLassoRegression = "LassoCV with Fast Caratheodory booster"


class RidgeRegressionMethods(BaseEnum):
    SkLearnLassoRegression = "Scikit-Learn's RidgeCV Method"
    BoostedLassoRegression = "RidgeCV with Fast Caratheodory booster"


class ElasticNetRegressionMethods(BaseEnum):
    SkLearnLassoRegression = "Scikit-Learn's ElasticNetCV Method"
    BoostedLassoRegression = "ElasticNetCV with Fast Caratheodory booster"


class NumpyDistribution(BaseEnum):
    IntelDistribution = "Intel's Numpy Distribution"
    CPythonDistribution = "CPython's Numpy distribution"
