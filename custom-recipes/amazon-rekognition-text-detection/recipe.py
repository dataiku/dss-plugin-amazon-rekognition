# -*- coding: utf-8 -*-
"""Text Detection recipe script"""

from typing import Dict, AnyStr
from ratelimit import limits, RateLimitException
from retry import retry

from plugin_params_loader import PluginParamsLoader
from amazon_rekognition_api_client import API_EXCEPTIONS, call_api_generic
from dku_io_utils import set_column_description
from api_parallelizer import api_parallelizer
from amazon_rekognition_api_formatting import TextDetectionAPIFormatter


# ==============================================================================
# SETUP
# ==============================================================================

plugin_params = PluginParamsLoader().validate_load_params()
column_prefix = "text_api"


@retry((RateLimitException, OSError), delay=plugin_params.api_quota_period, tries=5)
@limits(calls=plugin_params.api_quota_rate_limit, period=plugin_params.api_quota_period)
def call_api_text_detection(row: Dict, orientation_correction: bool) -> AnyStr:
    response_json = call_api_generic(
        row=row,
        orientation_correction=orientation_correction,
        api_client=plugin_params.api_client,
        api_client_method_name="detect_text",
        input_folder=plugin_params.input_folder,
        input_folder_is_s3=plugin_params.input_folder_is_s3,
        input_folder_bucket=plugin_params.input_folder_bucket,
        input_folder_root_path=plugin_params.input_folder_root_path,
    )
    return response_json


# ==============================================================================
# RUN
# ==============================================================================

# Call API in parallel
df = api_parallelizer(
    input_df=plugin_params.input_df,
    api_call_function=call_api_text_detection,
    api_exceptions=API_EXCEPTIONS,
    parallel_workers=plugin_params.parallel_workers,
    error_handling=plugin_params.error_handling,
    orientation_correction=plugin_params.orientation_correction,
    column_prefix=column_prefix,
)

# Format API results
api_formatter = TextDetectionAPIFormatter(
    input_df=plugin_params.input_df,
    minimum_score=plugin_params.minimum_score,
    orientation_correction=plugin_params.orientation_correction,
    input_folder=plugin_params.input_folder,
    error_handling=plugin_params.error_handling,
    parallel_workers=plugin_params.parallel_workers,
    column_prefix=column_prefix,
)
output_df = api_formatter.format_df(df)

# Write back results
plugin_params.output_dataset.write_with_schema(output_df)
set_column_description(
    output_dataset=plugin_params.output_dataset, column_description_dict=api_formatter.column_description_dict
)
if plugin_params.output_folder is not None:
    api_formatter.format_save_images(plugin_params.output_folder)
