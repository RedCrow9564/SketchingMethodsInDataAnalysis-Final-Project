from Infrastructure.enums import AlgorithmsType, LinearRegressionAlgorithms, LassoRegressionMethods,\
    RidgeRegressionMethods, ElasticNetRegressionMethods

_algorithms_type_to_algorithms = {
    AlgorithmsType.LinearRegression: LinearRegressionAlgorithms,
    AlgorithmsType.RidgeRegression: RidgeRegressionMethods,
    AlgorithmsType.LassoRegression: LassoRegressionMethods,
    AlgorithmsType.ElasticNetRegression: ElasticNetRegressionMethods
}


def get_methods(requested_algorithms_type: AlgorithmsType):
    return _algorithms_type_to_algorithms[requested_algorithms_type]
