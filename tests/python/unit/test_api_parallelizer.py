# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import json
from typing import AnyStr, Dict
from enum import Enum

import pandas as pd
from boto3.exceptions import Boto3Error

from api_parallelizer import api_parallelizer  # noqa


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = (Boto3Error, ValueError)
COLUMN_PREFIX = "test_api"
INPUT_COLUMN = "test_case"


class APICaseEnum(Enum):
    SUCCESS = {
        "test_api_response": '{"result": "Great success"}',
        "test_api_error_message": "",
        "test_api_error_type": "",
    }
    INVALID_INPUT = {
        "test_api_response": "",
        "test_api_error_message": "invalid literal for int() with base 10: 'invalid_integer'",
        "test_api_error_type": "ValueError",
    }
    API_FAILURE = {
        "test_api_response": "",
        "test_api_error_message": "",
        "test_api_error_type": "boto3.exceptions.Boto3Error",
    }


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def call_mock_api(row: Dict, api_function_param: int = 42) -> AnyStr:
    test_case = row.get(INPUT_COLUMN)
    response = {}
    if test_case == APICaseEnum.SUCCESS:
        response = {"result": "Great success"}
    elif test_case == APICaseEnum.INVALID_INPUT:
        try:
            response = {"result": int(api_function_param)}
        except ValueError as e:
            raise e
    elif test_case == APICaseEnum.API_FAILURE:
        raise Boto3Error
    return json.dumps(response)


def test_api_success():
    input_df = pd.DataFrame({INPUT_COLUMN: [APICaseEnum.SUCCESS]})
    df = api_parallelizer(
        input_df=input_df, api_call_function=call_mock_api, api_exceptions=API_EXCEPTIONS, column_prefix=COLUMN_PREFIX
    )
    output_dictionary = df.iloc[0, :].to_dict()
    expected_dictionary = APICaseEnum.SUCCESS.value
    for k in expected_dictionary:
        assert output_dictionary[k] == expected_dictionary[k]


def test_api_failure():
    input_df = pd.DataFrame({INPUT_COLUMN: [APICaseEnum.API_FAILURE]})
    df = api_parallelizer(
        input_df=input_df, api_call_function=call_mock_api, api_exceptions=API_EXCEPTIONS, column_prefix=COLUMN_PREFIX
    )
    output_dictionary = df.iloc[0, :].to_dict()
    expected_dictionary = APICaseEnum.API_FAILURE.value
    for k in expected_dictionary:
        assert output_dictionary[k] == expected_dictionary[k]


def test_invalid_input():
    input_df = pd.DataFrame({INPUT_COLUMN: [APICaseEnum.INVALID_INPUT]})
    df = api_parallelizer(
        input_df=input_df,
        api_call_function=call_mock_api,
        api_exceptions=API_EXCEPTIONS,
        column_prefix=COLUMN_PREFIX,
        api_function_param="invalid_integer",
    )
    output_dictionary = df.iloc[0, :].to_dict()
    expected_dictionary = APICaseEnum.INVALID_INPUT.value
    for k in expected_dictionary:
        assert output_dictionary[k] == expected_dictionary[k]
