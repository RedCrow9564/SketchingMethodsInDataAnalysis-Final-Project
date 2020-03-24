from numpy.linalg import norm
from Infrastructure.utils import ex, create_factory, Dict, List, Callable, Vector, Matrix, DataLog, measure_time
from Infrastructure.enums import ExperimentType, LogFields


@ex.capture(prefix="run_time_experiments_config")
def _run_time_experiment(compared_solvers: List, data_matrix: Matrix, output_samples: Vector,
                         results_path: str, run_time_compared_data_sizes: List):
    for solver in compared_solvers:
        data_log = DataLog([LogFields.DataSize, LogFields.Coefficients, LogFields.Residuals,
                            LogFields.DurationInSeconds, LogFields.AtTimesErrors])
        coefficients_list = list()
        residuals_list = list()
        transpose_errors_list = list()
        durations_list = list()
        solver_with_time_measure: Callable = measure_time(solver)

        for data_size in run_time_compared_data_sizes:
            partial_data: Matrix = data_matrix[:data_size, :]
            coefficients, residuals, duration = solver_with_time_measure(partial_data, output_samples[:data_size])

            coefficients_list.append(coefficients.tolist())
            residuals_list.append(norm(residuals))
            transpose_errors_list.append(norm(partial_data.T.dot(residuals)))
            durations_list.append(duration)
            print(f'solver name={solver.__name__}, data size={data_size}, duration={duration}')

        data_log.append(LogFields.DataSize, list(run_time_compared_data_sizes))
        data_log.append(LogFields.Residuals, residuals_list)
        data_log.append(LogFields.AtTimesErrors, transpose_errors_list)
        data_log.append(LogFields.Coefficients, coefficients_list)
        data_log.append(LogFields.DurationInSeconds, durations_list)
        data_log.save_log(solver.__name__ + ".csv", results_path)


@ex.capture(prefix="number_of_alphas_experiments_config")
def _number_of_alphas_experiment(compared_solvers: List, data_matrix: Matrix, output_samples: Vector,
                                 results_path: str, alphas_range: List):
    for solver in compared_solvers:
        data_log = DataLog([LogFields.Coefficients, LogFields.Residuals, LogFields.DurationInSeconds,
                            LogFields.AlphasCount])
        coefficients_list = list()
        residuals_list = list()
        durations_list = list()
        solver_with_time_measure: Callable = measure_time(solver)

        for current_alphas in alphas_range:
            coefficients, residuals, duration = solver_with_time_measure(data_matrix, output_samples,
                                                                         n_alphas=current_alphas)

            coefficients_list.append(coefficients.tolist())
            residuals_list.append(norm(residuals))
            durations_list.append(duration)
            print(f'solver name={solver.__name__}, total alphas={current_alphas}, duration={duration}')

        data_log.append(LogFields.AlphasCount, list(alphas_range))
        data_log.append(LogFields.Residuals, residuals_list)
        data_log.append(LogFields.Coefficients, coefficients_list)
        data_log.append(LogFields.DurationInSeconds, durations_list)
        data_log.save_log(solver.__name__ + ".csv", results_path)


_experiment_type_to_method: Dict = {
    ExperimentType.RunTimeExperiment: _run_time_experiment,
    ExperimentType.NumberOfAlphasExperiment: _number_of_alphas_experiment
}

create_experiment: Callable = create_factory(_experiment_type_to_method, are_methods=True)
