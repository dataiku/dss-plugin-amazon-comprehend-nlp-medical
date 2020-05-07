# -*- coding: utf-8 -*-
import json
from typing import List, Dict, AnyStr, Union

from retry import retry
from ratelimit import limits, RateLimitException

import dataiku

from plugin_io_utils import (
    ErrorHandlingEnum,
    validate_column_input,
    set_column_description,
)
from api_parallelizer import api_parallelizer
from dataiku.customrecipe import (
    get_recipe_config,
    get_input_names_for_role,
    get_output_names_for_role,
)
from api_formatting import (
    EntityTypeEnum,
    get_client,
    NamedEntityRecognitionAPIFormatter,
)


# ==============================================================================
# SETUP
# ==============================================================================

api_configuration_preset = get_recipe_config().get("api_configuration_preset")
api_quota_rate_limit = api_configuration_preset.get("api_quota_rate_limit")
api_quota_period = api_configuration_preset.get("api_quota_period")
parallel_workers = api_configuration_preset.get("parallel_workers")
batch_size = api_configuration_preset.get("batch_size")
text_column = get_recipe_config().get("text_column")
text_language = get_recipe_config().get("language")
language_column = get_recipe_config().get("language_column")
entity_types = [EntityTypeEnum[i] for i in get_recipe_config().get("entity_types", [])]
error_handling = ErrorHandlingEnum[get_recipe_config().get("error_handling")]

input_dataset_name = get_input_names_for_role("input_dataset")[0]
input_dataset = dataiku.Dataset(input_dataset_name)
input_schema = input_dataset.read_schema()
input_columns_names = [col["name"] for col in input_schema]

output_dataset_name = get_output_names_for_role("output_dataset")[0]
output_dataset = dataiku.Dataset(output_dataset_name)

validate_column_input(text_column, input_columns_names)

api_support_batch = True
if text_language == "language_column":
    api_support_batch = False
    validate_column_input(language_column, input_columns_names)

input_df = input_dataset.get_dataframe()
client = get_client(api_configuration_preset, "comprehend")
column_prefix = "entity_api"


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=api_quota_period, tries=5)
@limits(calls=api_quota_rate_limit, period=api_quota_period)
def call_api_named_entity_recognition(
    text_column: AnyStr,
    text_language: AnyStr,
    language_column: AnyStr = None,
    row: Dict = None,
    batch: List[Dict] = None,
) -> List[Union[Dict, AnyStr]]:
    if text_language == "language_column":
        # Cannot use batch as language may be different for each row
        text = row[text_column]
        language_code = row[language_column]
        empty_conditions = [
            not (isinstance(text, str)),
            not (isinstance(language_code, str)),
            str(text).strip() == "",
            str(language_code).strip() == "",
        ]
        if any(empty_conditions):
            return ""
        response = client.detect_entities(Text=text, LanguageCode=language_code)
        return json.dumps(response)
    else:
        text_list = [str(r.get(text_column, "")).strip() for r in batch]
        responses = client.batch_detect_entities(
            TextList=text_list, LanguageCode=text_language
        )
        return responses


df = api_parallelizer(
    input_df=input_df,
    api_call_function=call_api_named_entity_recognition,
    text_column=text_column,
    text_language=text_language,
    language_column=language_column,
    parallel_workers=parallel_workers,
    api_support_batch=api_support_batch,
    batch_size=batch_size,
    error_handling=error_handling,
    column_prefix=column_prefix,
)

api_formatter = NamedEntityRecognitionAPIFormatter(
    input_df=input_df,
    entity_types=entity_types,
    column_prefix=column_prefix,
    error_handling=error_handling,
)
output_df = api_formatter.format_df(df)

output_dataset.write_with_schema(output_df)
set_column_description(
    input_dataset=input_dataset,
    output_dataset=output_dataset,
    column_description_dict=api_formatter.column_description_dict,
)
