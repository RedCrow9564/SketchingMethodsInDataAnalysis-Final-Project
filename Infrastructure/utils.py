from copy import deepcopy
from typing import List, Dict, Callable, Tuple, ClassVar, Iterable, Mapping, Union, Iterator
from nptyping import Array
import inspect
from multiprocessing import Pool
from sacred import Experiment

# Defining the "sacred" experiment object.
ex = Experiment(name="Initializing project", interactive=False)


# Naming data types for type hinting.
Number = Union[int, float]
Scalar = Union[Number, Array[float, 1, 1], Array[int, 1, 1]]
RowVector = Union[List[Scalar], Array[float, 1, ...], Array[int, 1, ...], Scalar]
ColumnVector = Union[List[Scalar], Array[float, ..., 1], Array[int, ..., 1], Scalar]
Vector = Union[RowVector, ColumnVector]
Matrix = Union[List[Vector], Array[float], Array[int], Vector, Scalar]
StaticField = ClassVar


class _MetaEnum(type):
    def __iter__(self) -> Iterator:
        # noinspection PyUnresolvedReferences
        return self.enum_iter()

    def __contains__(self, item) -> bool:
        # noinspection PyUnresolvedReferences
        return self.enum_contains(item)


class BaseEnum(metaclass=_MetaEnum):

    @classmethod
    def enum_iter(cls) -> Iterator:
        return iter(cls.get_all_values())

    @classmethod
    def enum_contains(cls, item) -> bool:
        return item in cls.get_all_values()

    @classmethod
    def get_all_values(cls) -> List:
        all_attributes: List = inspect.getmembers(cls, lambda a: not inspect.ismethod(a))
        all_attributes = [value for name, value in all_attributes if not (name.startswith('__') or name.endswith('__'))]
        return all_attributes


def get_user_defined_classes(file_scope):
    user_classes = [x for x in dir(file_scope) if not x.startswith('__') or not x.endswith('__')]
    return user_classes


def create_factory(possibilities_dict: Dict[str, Callable]) -> Callable:
    """
    A generic method for creating factories for the entire project.
    :param possibilities_dict: The dictionary which maps object types (as strings!) and returns the
     relevant class constructors.
    :return: The factory function for the given classes mapping.
    """
    def factory_func(requested_object_type: str):
        if requested_object_type not in possibilities_dict:
            raise ValueError("Object type {0} is NOT supported".format(requested_object_type))
        else:
            return possibilities_dict[requested_object_type]()

    return factory_func


def compute_parallel(func: Callable, arguments):
    pool = Pool()
    results = pool.starmap(func, arguments)
    pool.close()
    return results


class DataLog:
    def __init__(self, log_fields):
        # TODO: Consider implementing the data-log using pandas.
        self.data = dict()
        for log_field in log_fields:
            self.data[log_field] = []

    def append(self, data_type: str, value: Vector) -> None:
        self.data[data_type].append(deepcopy(value))

    def append_dict(self, data_dict: Dict) -> None:
        for log_field, data_value in data_dict.items():
            self.append(log_field, data_value)

    def save_log(self):
        ex.info["Experiment Log"] = self.data
