import matplotlib.pyplot as plt
from matplotlib import rc
import os
import pandas as pd
import numpy as np
from Infrastructure.utils import ex, DataLog, List, Dict, Vector, Matrix, is_empty
from Infrastructure.enums import AlgorithmsType, DatabaseType, NumpyDistribution, ExperimentType, LogFields


class GraphManager:
    def __init__(self, super_title: str, graphs_count: int):
        self._super_title: str = super_title
        self._graphs_count = graphs_count
        self._colors = ["b", "g", "y", "r"]
        self._colors = {
            "d=3": "b", "d=5": "g", "d=7": "r"
        }
        if graphs_count % 2 == 0:
            self._rows = graphs_count / 2
            self._cols = 2
        else:
            self._rows = 1
            self._cols = graphs_count
        self._current_col = 0
        self._current_graph = 1
        rc('text', usetex=True)
        rc('font', family='serif')

    def add_plot(self, x_values: Vector, x_label: str, data_values: Matrix, data_label: str, plot_title: str,
                 legends: List[str], marker=".", linestyle="-") -> None:
        plt.subplot(self._rows, self._cols, self._current_graph)

        for one_legend, experiment_data in zip(legends, data_values):
            if "BOOST" in one_legend:
                if "d=3" in one_legend:
                    plt.plot(x_values[::2], experiment_data[::2], marker=marker, linestyle=":", color="b")
                elif "d=5" in one_legend:
                    plt.plot(x_values[::2], experiment_data[::2], marker=marker, linestyle=":", color="g")
                elif "d=7" in one_legend:
                    plt.plot(x_values[::2], experiment_data[::2], marker=marker, linestyle=":", color="r")
            else:
                if "d=3" in one_legend:
                    plt.plot(x_values[::2], experiment_data[::2], marker=marker, linestyle="-", color="b")
                elif "d=5" in one_legend:
                    plt.plot(x_values[::2], experiment_data[::2], marker=marker, linestyle="-", color="g")
                elif "d=7" in one_legend:
                    plt.plot(x_values[::2], experiment_data[::2], marker=marker, linestyle="-", color="r")

        plt.legend(legends, fontsize=6, loc="upper left")
        plt.xlabel("\\textit{" + x_label + "}", fontsize=12)
        if self._current_graph == 1:
            plt.ylabel("\\textit{" + data_label + "}", fontsize=12)
        plt.title("\\textit{" + plot_title + "}", fontsize=12)  # y=1.08)
        plt.grid(True)
        #plt.axes().set_aspect('equal', 'datalim')
        self._current_graph += 1

    def add_run_time_plot(self, data_size_values: Vector, data_values: Matrix, plot_title, legends: List[str],
                          linestyle):
        self.add_plot(data_size_values, "Data size n", data_values, "Computation time [seconds]", plot_title, legends,
                      6, linestyle)

    def add_accuracy_histogram(self):
        pass

    def show(self) -> None:
        #plt.suptitle(self._super_title)
        plt.show()

    @ex.capture
    def save_plot(self, graphs_directory) -> None:
        file_path: str = os.path.join(graphs_directory, self._super_title)
        plt.savefig(file_path)
        ex.add_artifact("{0}.png".format(self._super_title))
        os.remove(file_path)


@ex.config
def config():
    """ Config section

    This function contains all possible configuration for all experiments. Full details on each configuration values
    can be found in Infrastructure/enums.py.
    """

    compared_algorithms_type: AlgorithmsType = AlgorithmsType.LinearRegression
    numpy_distribution: NumpyDistribution = NumpyDistribution.CPythonDistribution
    used_database: DatabaseType = DatabaseType.Synthetic
    experiment_type: ExperimentType = ExperimentType.RunTimeExperiment
    cross_validation_folds: int = 3
    n_alphas: int = 100

    run_time_experiments_config: Dict[str, range] = {
        "run_time_compared_data_sizes": range(100000, 2400000, 100000)
    }
    synthetic_data_config: Dict[str, int] = {
        "data_size": 230,
        "features_num": 10
    }
    number_of_alphas_experiments_config: Dict[str, range] = {
        "alphas_range": range(1, 201, 20)
    }
    results_path: str = os.path.join(
        r'..', 'Results', 'Run-Time comparison', 'different dimensions d')

    clusters_count: int = 2
    elastic_net_factor: float = 0.5  # Rho factor in Elastic-Net regularization.


def _load_all_csv(folder_path: str, field_in_table: str) -> (Matrix, List):
    files_names = list()
    results = list()

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            experiment_data: Matrix = pd.read_csv(os.path.join(folder_path, file))
            results.append(experiment_data[field_in_table].values.reshape((1, -1)))
            files_names.append("\\textit{" + file[:-4].replace("_", " ") + "}")

    if is_empty(results):
        return list(), list()
    else:
        return np.vstack(results), files_names


@ex.automain
def plot_results(results_path, experiment_type, run_time_experiments_config, numpy_distribution,
                 number_of_alphas_experiments_config) -> None:
    sub_folders: List = next(os.walk(results_path))[1]
    folders_count = len(sub_folders)

    if experiment_type == ExperimentType.RunTimeExperiment:
        g = GraphManager(f'Run-time comparison of solvers over synthetic data, {numpy_distribution}', folders_count)
        for sub_folder in sub_folders:
            run_times, legends = _load_all_csv(os.path.join(results_path, sub_folder), LogFields.DurationInSeconds)
            compared_data_sizes: Vector = list(run_time_experiments_config["run_time_compared_data_sizes"])
            if not is_empty(legends):
                g.add_run_time_plot(compared_data_sizes, run_times, sub_folder, legends, "-")

    elif experiment_type == ExperimentType.AccuracyExperiment:
        g = GraphManager(f'Accuracy comparison of solvers over synthetic data, {numpy_distribution}', folders_count)
        for sub_folder in sub_folders:
            residuals, legends = _load_all_csv(os.path.join(results_path, sub_folder), LogFields.Residuals)

            # Remove the residuals for standard SVD solver
            compared_residuals: Vector = residuals
            #if not is_empty(legends):
            #    g.add_run_time_plot(compared_data_sizes, run_times, sub_folder, legends)

    elif experiment_type == ExperimentType.NumberOfAlphasExperiment:
        g = GraphManager(f'Run-Time comparison for increasing alpha terms, {numpy_distribution}', folders_count)
        for sub_folder in sub_folders:
            run_times, legends = _load_all_csv(os.path.join(results_path, sub_folder), LogFields.DurationInSeconds)
            compared_data_sizes: Vector = list(number_of_alphas_experiments_config["alphas_range"])
            if not is_empty(legends):
                g.add_run_time_plot(compared_data_sizes, run_times, sub_folder, legends, "-")

    g.show()
