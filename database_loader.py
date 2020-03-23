# -*- coding: utf-8 -*-
""" The data management module

This module handles the fetching of the data from the local resources path, given in the configuration and arranging it
for our purposes of estimations. See the example for fetching the 3D Road Network database.

Example:
-------
    get_data(DatabaseType.ThreeDRoadNetwork) - Creating the standard Numpy solver for Linear-Regression.

"""

from numpy.random import rand
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import os
from Infrastructure.enums import DatabaseType
from Infrastructure.utils import ex, Dict, ColumnVector, Matrix, create_factory, Callable


@ex.capture(prefix="synthetic_data_config")
def _get_synthetic_data(data_size: int, features_num: int) -> (Matrix, ColumnVector):
    """
    A method which creates a random matrix of size data_size x features_num and a random 1 x data_size vector.

    Attributes:
    -----------
        data_size(int): The input data matrix nxd.
        features_num(int): The output for the given inputs, nx1.

    Returns:
    --------
        A random size data_size x features_num Matrix and a random 1 x data_size ColumnVector.

    """
    return rand(data_size, features_num), rand(data_size)


@ex.capture
def _get_3d_road_network_data(resources_path: str) -> (Matrix, ColumnVector):
    """
    A method which loads the 3D Road Network database from the given local path.
    See https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+(North+Jutland,+Denmark)

    Attributes:
    -----------
        resources_path(str): The path in which this database is stored in.

    Returns:
    --------
        A Matrix of the features, besides the height, and a ColumnVector of the height feature.

    """
    data: Matrix = scale(np.loadtxt(os.path.join(resources_path, "3D_spatial_network.txt"), delimiter=','))
    output_samples: ColumnVector = data[:, -1]
    data_matrix: Matrix = data[:, 1:3]
    return data_matrix, output_samples


@ex.capture
def _get_house_sales_in_king_county_data(resources_path: str) -> (Matrix, ColumnVector):
    """
    A method which loads the House sales in King County database from the given local path.
    See https://www.kaggle.com/harlfoxem/housesalesprediction

    Attributes:
    -----------
        resources_path(str): The path in which this database is stored in.

    Returns:
    --------
        A Matrix of the features, besides the price, and a ColumnVector of the price feature.

    """
    df = pd.read_csv(os.path.join(resources_path, "kc_house_data.csv"))
    output_samples: ColumnVector = scale(df["price"].to_numpy())
    data_matrix: Matrix = scale(df.drop(columns=["id", "price", "date"]).to_numpy())
    return data_matrix, output_samples


@ex.capture
def _get_household_electric_power_consumption_data(resources_path: str) -> (Matrix, ColumnVector):
    """
    A method which loads the Household electric power consumption database from the given local path.
    See https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption.

    Attributes:
    -----------
        resources_path(str): The path in which this database is stored in.

    Returns:
    --------
        A Matrix of the features, besides the voltage, and a ColumnVector of the voltage feature.

    """
    df = pd.read_csv(os.path.join(resources_path, "household_power_consumption.txt"), sep=';', na_values="?")
    df.dropna(axis=0, inplace=True)
    output_samples: ColumnVector = scale(pd.to_numeric(df["Voltage"]).to_numpy())
    data_matrix: Matrix = scale(df[["Global_active_power", "Global_reactive_power"]].astype("float64").to_numpy())
    return data_matrix, output_samples


# A private dictionary used to create the method "get_data"
_database_type_to_function: Dict[str, Callable] = {
    DatabaseType.Synthetic: _get_synthetic_data,
    DatabaseType.ThreeDRoadNetwork: _get_3d_road_network_data,
    DatabaseType.HouseSalesInKingCounty: _get_house_sales_in_king_county_data,
    DatabaseType.IndividualHouseholdElectricPowerConsumption: _get_household_electric_power_consumption_data
}

# The public method which fetches the data loading methods.
get_data: Callable = create_factory(_database_type_to_function)
