import matplotlib.pyplot as plt
from matplotlib import rc
import os
from Infrastructure.utils import ex, DataLog, List, Vector


class GraphManager:
    def __init__(self, super_title: str, graphs_count: int):
        self._super_title: str = super_title
        self._graphs_count = graphs_count
        if graphs_count % 2 == 0:
            self._rows = graphs_count / 2
            self._cols = 2
        else:
            self._rows = graphs_count
            self._cols = 1
        self._current_col = 0
        self._current_graph = 1
        rc('text', usetex=True)
        rc('font', family='serif')

    def add_plot(self, time_values: Vector, data_values: Vector, data_label: str, plot_title: str,
                 legends: List[str]) -> None:
        plt.subplot(self._rows, self._cols, self._current_graph)
        plt.plot(time_values, data_values)
        plt.legend(legends)
        plt.xlabel('\\textit{Time}', fontsize=16)
        plt.ylabel('\\textit{' + data_label + '}', fontsize=16)
        plt.title(plot_title, y=1.08)
        self._current_graph += 1

    def show(self) -> None:
        #plt.suptitle(self._super_title)
        plt.show()

    @ex.capture
    def save_plot(self, graphs_directory) -> None:
        file_path: str = os.path.join(graphs_directory, self._super_title)
        plt.savefig(file_path)
        ex.add_artifact("{0}.png".format(self._super_title))
        os.remove(file_path)


def plot_results(data_log: DataLog) -> None:
    pass
