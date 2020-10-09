# -*- coding: utf-8 -*-
"""Module with read/write utility functions which are *not* based on the Dataiku API"""

import logging
import json
from enum import Enum
from typing import AnyStr, List, NamedTuple, Dict
from collections import OrderedDict, namedtuple

import pandas as pd


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

COLUMN_PREFIX = "api"
API_COLUMN_NAMES_DESCRIPTION_DICT = OrderedDict(
    [
        ("response", "Raw response from the API in JSON format"),
        ("error_message", "Error message from the API"),
        ("error_type", "Error type (module and class name)"),
        ("error_raw", "Raw error from the API"),
    ]
)

ApiColumnNameTuple = namedtuple("ApiColumnNameTuple", API_COLUMN_NAMES_DESCRIPTION_DICT.keys())


class ErrorHandlingEnum(Enum):
    LOG = "Log"
    FAIL = "Fail"


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def generate_unique(name: AnyStr, existing_names: List, prefix: AnyStr = COLUMN_PREFIX) -> AnyStr:
    """
    Generate a unique name among existing ones by suffixing a number. Can also add an optional prefix.
    """
    if prefix is not None:
        new_name = prefix + "_" + name
    else:
        new_name = name
    for j in range(1, 1001):
        if new_name not in existing_names:
            return new_name
        new_name = name + "_{}".format(j)
    raise Exception("Failed to generated a unique name")


def build_unique_column_names(existing_names: List[AnyStr], column_prefix: AnyStr = COLUMN_PREFIX) -> NamedTuple:
    """
    Helper function to the "api_parallelizer" main function.
    Initializes a named tuple of column names from ApiColumnNameTuple, ensure columns are unique.
    """
    api_column_names = ApiColumnNameTuple(
        *[generate_unique(k, existing_names, column_prefix) for k in ApiColumnNameTuple._fields]
    )
    return api_column_names


def validate_column_input(column_name: AnyStr, column_list: List[AnyStr]) -> None:
    """
    Validate that user input for column parameter is valid.
    """
    if column_name is None or len(column_name) == 0:
        raise ValueError("You must specify a valid column name.")
    if column_name not in column_list:
        raise ValueError("Column '{}' is not present in the input dataset.".format(column_name))


def safe_json_loads(
    str_to_check: AnyStr, error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG, verbose: bool = False,
) -> Dict:
    """
    Wrap json.loads with an additional parameter to handle errors:
    - 'FAIL' to use json.loads, which throws an exception on invalid data
    - 'LOG' to try json.loads and return an empty dict if data is invalid
    """
    if error_handling == ErrorHandlingEnum.FAIL:
        output = json.loads(str_to_check)
    else:
        try:
            output = json.loads(str_to_check)
        except (TypeError, ValueError):
            if verbose:
                logging.warning("Invalid JSON: '" + str(str_to_check) + "'")
            output = {}
    return output


def move_api_columns_to_end(
    df: pd.DataFrame, api_column_names: NamedTuple, error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG
) -> pd.DataFrame:
    """
    Move non-human-readable API columns to the end of the dataframe
    """
    api_column_names_dict = api_column_names._asdict()
    if error_handling == ErrorHandlingEnum.FAIL:
        api_column_names_dict.pop("error_message", None)
        api_column_names_dict.pop("error_type", None)
    if not any(["error_raw" in k for k in df.keys()]):
        api_column_names_dict.pop("error_raw", None)
    cols = [c for c in df.keys() if c not in api_column_names_dict.values()]
    new_cols = cols + list(api_column_names_dict.values())
    df = df.reindex(columns=new_cols)
    return df
