from numpy.linalg import norm
import time
from Infrastructure.utils import ex, create_factory, Dict, List, Callable, Vector, Matrix, DataLog, measure_time
from Infrastructure.enums import ExperimentType, LogFields


@ex.capture()
def _run_time_experiment(compared_solvers: List, data_matrix: Matrix, output_samples: Vector,
                         run_time_experiments_config):
    all_logs: Dict = dict()
    run_time_compared_data_sizes: List = run_time_experiments_config["run_time_compared_data_sizes"]
    calc_transpose_dot_residuals: bool = run_time_experiments_config["calc_transpose_dot_residuals"]

    for solver in compared_solvers:
        error_raised: bool = False
        log_fields: List = [LogFields.DataSize, LogFields.Coefficients, LogFields.Residuals,
                            LogFields.DurationInSeconds]
        if calc_transpose_dot_residuals:
            log_fields.append(LogFields.AtTimesErrors)

        data_log = DataLog(log_fields)
        coefficients_list = list()
        residuals_list = list()
        transpose_errors_list = list()
        durations_list = list()

        for data_size in run_time_compared_data_sizes:
            try:
                partial_data: Matrix = data_matrix[:data_size, :]
                partial_samples: Vector = output_samples[:data_size]
                solver_obj = solver(partial_data, partial_samples)
                coefficients, duration = measure_time(solver_obj.fit)()
                residuals: Vector = solver_obj.calc_residuals()

                coefficients_list.append(coefficients.tolist())
                residuals_list.append(norm(residuals))
                if calc_transpose_dot_residuals:
                    transpose_errors_list.append(norm(partial_data.T.dot(residuals)))
                durations_list.append(duration)
                print(f'solver name={solver.__name__}, data size={data_size}, duration={duration}')

            except IOError:
                error_raised = True
                continue

        if not error_raised:
            data_log.append(LogFields.DataSize, list(run_time_compared_data_sizes))
            data_log.append(LogFields.Residuals, residuals_list)
            if calc_transpose_dot_residuals:
                data_log.append(LogFields.AtTimesErrors, transpose_errors_list)
            data_log.append(LogFields.Coefficients, coefficients_list)
            data_log.append(LogFields.DurationInSeconds, durations_list)
            all_logs[solver.__name__] = data_log

    return all_logs


@ex.capture(prefix="number_of_alphas_experiments_config")
def _number_of_alphas_experiment(compared_solvers: List, data_matrix: Matrix, output_samples: Vector,
                                 alphas_range: List):
    all_logs: Dict = dict()
    for solver in compared_solvers:
        error_raised: bool = False
        data_log = DataLog([LogFields.Coefficients, LogFields.Residuals, LogFields.DurationInSeconds,
                            LogFields.AlphasCount])
        coefficients_list = list()
        residuals_list = list()
        durations_list = list()

        for current_alphas in alphas_range:
            try:
                solver_obj = solver(data_matrix, output_samples, n_alphas=current_alphas)
                coefficients, duration = measure_time(solver_obj.fit)()
                residuals: Vector = solver_obj.calc_residuals()

                coefficients_list.append(coefficients.tolist())
                residuals_list.append(norm(residuals))
                durations_list.append(duration)
                print(f'solver name={solver.__name__}, total alphas={current_alphas}, duration={duration}')

            except IOError:
                error_raised = True
                continue

        if not error_raised:
            data_log.append(LogFields.AlphasCount, list(alphas_range))
            data_log.append(LogFields.Residuals, residuals_list)
            data_log.append(LogFields.Coefficients, coefficients_list)
            data_log.append(LogFields.DurationInSeconds, durations_list)
            all_logs[solver.__name__] = data_log

    return all_logs


_experiment_type_to_method: Dict = {
    ExperimentType.RunTimeExperiment: _run_time_experiment,
    ExperimentType.NumberOfAlphasExperiment: _number_of_alphas_experiment
}

create_experiment: Callable = create_factory(_experiment_type_to_method, are_methods=True)
