# -*- coding: utf-8 -*-
import json
from typing import Dict, AnyStr

from retry import retry
from ratelimit import limits, RateLimitException

import dataiku
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role

from plugin_io_utils import ErrorHandlingEnum, validate_column_input
from dku_io_utils import set_column_description
from amazon_comprehend_medical_api_formatting import MedicalPhiAPIFormatter
from amazon_comprehend_medical_api_client import API_EXCEPTIONS, get_client
from api_parallelizer import api_parallelizer


# ==============================================================================
# SETUP
# ==============================================================================

api_configuration_preset = get_recipe_config().get("api_configuration_preset")
api_quota_rate_limit = api_configuration_preset.get("api_quota_rate_limit")
api_quota_period = api_configuration_preset.get("api_quota_period")
parallel_workers = api_configuration_preset.get("parallel_workers")
text_column = get_recipe_config().get("text_column")
minimum_score = float(get_recipe_config().get("minimum_score", 0))
if minimum_score < 0 or minimum_score > 1:
    raise ValueError("Minimum confidence score must be between 0 and 1")
error_handling = ErrorHandlingEnum[get_recipe_config().get("error_handling")]

input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col["name"] for col in input_schema]
validate_column_input(text_column, input_columns_names)

output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

input_df = input_dataset.get_dataframe(infer_with_pandas=False)
client = get_client(api_configuration_preset)
column_prefix = "medical_phi_api"


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=api_quota_period, tries=5)
@limits(calls=api_quota_rate_limit, period=api_quota_period)
def call_api_medical_phi_extraction(row: Dict, text_column: AnyStr) -> Dict:
    text = row[text_column]
    if not isinstance(text, str) or str(text).strip() == "":
        return ""
    responses = client.detect_phi(Text=text)
    return json.dumps(responses)


df = api_parallelizer(
    input_df=input_df,
    api_call_function=call_api_medical_phi_extraction,
    api_exceptions=API_EXCEPTIONS,
    text_column=text_column,
    parallel_workers=parallel_workers,
    error_handling=error_handling,
    column_prefix=column_prefix,
)

api_formatter = MedicalPhiAPIFormatter(
    input_df=input_df, minimum_score=minimum_score, column_prefix=column_prefix, error_handling=error_handling,
)
output_df = api_formatter.format_df(df)

output_dataset.write_with_schema(output_df)
set_column_description(
    input_dataset=input_dataset,
    output_dataset=output_dataset,
    column_description_dict=api_formatter.column_description_dict,
)
