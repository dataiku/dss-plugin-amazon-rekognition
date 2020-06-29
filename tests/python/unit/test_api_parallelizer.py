# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import sys
import json
import os
from pathlib import Path
from typing import AnyStr, Dict
from enum import Enum

import pandas as pd
from boto3.exceptions import Boto3Error

# Add python-lib to the path to enable exec outside of DSS
plugin_root_path = Path(__file__).parents[3]
sys.path.append(os.path.join(plugin_root_path, "python-lib"))
from api_parallelizer import api_parallelizer  # noqa


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = (Boto3Error, ValueError)
COLUMN_PREFIX = "test_api"


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


INPUT_COLUMN = "test_case"
TEST_INPUT_DF = pd.DataFrame({INPUT_COLUMN: list(APICaseEnum)})

# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def call_test_api(row: Dict, api_function_param: int) -> AnyStr:
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


def test_api_parallelizer():
    df = api_parallelizer(
        input_df=TEST_INPUT_DF,
        api_call_function=call_test_api,
        api_exceptions=API_EXCEPTIONS,
        column_prefix=COLUMN_PREFIX,
        api_function_param="invalid_integer",
    )
    api_columns = df.keys()[1:]
    for test_case in APICaseEnum:
        output_dict = df.loc[df[INPUT_COLUMN] == test_case, api_columns].to_dict(orient="records")[0]
        assert output_dict == test_case.value
