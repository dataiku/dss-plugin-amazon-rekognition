# -*- coding: utf-8 -*-
from typing import Dict, AnyStr
from ratelimit import limits, RateLimitException
from retry import retry

from plugin_params_loader import PluginParamsLoader
from amazon_rekognition_api_client import API_EXCEPTIONS, call_api_generic
from dku_io_utils import set_column_description
from api_parallelizer import api_parallelizer
from amazon_rekognition_api_formatting import UnsafeContentAPIFormatter


# ==============================================================================
# SETUP
# ==============================================================================

plugin_params = PluginParamsLoader().load_validate_params()
column_prefix = "moderation_api"


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=plugin_params.api_quota_period, tries=5)
@limits(calls=plugin_params.api_quota_rate_limit, period=plugin_params.api_quota_period)
def call_api_moderation(row: Dict, minimum_score: int) -> AnyStr:
    response_json = call_api_generic(
        row=row,
        minimum_score=minimum_score,
        api_client=plugin_params.api_client,
        api_client_method_name="detect_moderation_labels",
        input_folder=plugin_params.input_folder,
        input_folder_is_s3=plugin_params.input_folder_is_s3,
        input_folder_bucket=plugin_params.input_folder_bucket,
        input_folder_root_path=plugin_params.input_folder_root_path,
    )
    return response_json


df = api_parallelizer(
    input_df=plugin_params.input_df,
    api_call_function=call_api_moderation,
    api_exceptions=API_EXCEPTIONS,
    parallel_workers=plugin_params.parallel_workers,
    error_handling=plugin_params.error_handling,
    minimum_score=plugin_params.minimum_score,
    column_prefix=column_prefix,
)

api_formatter = UnsafeContentAPIFormatter(
    input_df=plugin_params.input_df,
    category_level=plugin_params.unsafe_content_category_level,
    content_categories_top_level=plugin_params.unsafe_content_categories_top_level,
    content_categories_second_level=plugin_params.unsafe_content_categories_second_level,
    error_handling=plugin_params.error_handling,
    column_prefix=column_prefix,
)
output_df = api_formatter.format_df(df)

plugin_params.output_dataset.write_with_schema(output_df)
set_column_description(
    output_dataset=plugin_params.output_dataset, column_description_dict=api_formatter.column_description_dict
)
