# -*- coding: utf-8 -*-
import json
from typing import Dict, AnyStr
from ratelimit import limits, RateLimitException
from retry import retry

import dataiku
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role

from amazon_rekognition_api_client import get_client
from plugin_io_utils import ErrorHandlingEnum, set_column_description
from api_parallelizer import api_parallelizer
from amazon_rekognition_api_formatting import NamedEntityRecognitionAPIFormatter


# ==============================================================================
# SETUP
# ==============================================================================

recipe_config = get_recipe_config()
api_configuration_preset = recipe_config.get("api_configuration_preset")
api_quota_rate_limit = api_configuration_preset.get("api_quota_rate_limit")
api_quota_period = api_configuration_preset.get("api_quota_period")
parallel_workers = api_configuration_preset.get("parallel_workers")
num_objects = int(recipe_config.get("num_objects", 1))
if num_objects < 1:
    raise ValueError("Number of objects must be greater than 1")
minimum_score = float(recipe_config.get("minimum_score", 0))
if minimum_score < 0 or minimum_score > 1:
    raise ValueError("Minimum confidence score must be between 0 and 1")
error_handling = ErrorHandlingEnum[recipe_config.get("error_handling")]

client = get_client(api_configuration_preset)
column_prefix = "object_api"

input_df = input_dataset.get_dataframe()


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=api_quota_period, tries=5)
@limits(calls=api_quota_rate_limit, period=api_quota_period)
def call_api_object_detection(row: Dict, num_objects: int, minimum_score: float) -> AnyStr:
    text = row[text_column]
    if not isinstance(text, str) or str(text).strip() == "":
        return ""
    responses = client.detect_entities_v2(Text=text)
    return json.dumps(responses)


df = api_parallelizer(
    input_df=input_df,
    api_call_function=call_api_named_entity_recognition,
    api_exceptions=API_EXCEPTIONS,
    column_prefix=column_prefix,
    parallel_workers=parallel_workers,
    error_handling=error_handling,
    text_column=text_column,
    text_language=text_language,
    entity_sentiment=entity_sentiment,
)

api_formatter = NamedEntityRecognitionAPIFormatter(
    input_df=input_df,
    column_prefix=column_prefix,
    entity_types=entity_types,
    minimum_score=minimum_score,
    error_handling=error_handling,
)
output_df = api_formatter.format_df(df)

output_dataset.write_with_schema(output_df)
set_column_description(
    input_dataset=input_dataset,
    output_dataset=output_dataset,
    column_description_dict=api_formatter.column_description_dict,
)
