# -*- coding: utf-8 -*-
import json
from typing import Dict, AnyStr
from ratelimit import limits, RateLimitException
from retry import retry


from plugin_params_loader import PluginParamsLoader
from amazon_rekognition_api_client import API_EXCEPTIONS
from plugin_io_utils import IMAGE_PATH_COLUMN
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
    image_path = row.get(IMAGE_PATH_COLUMN)
    if plugin_params.input_folder_is_s3:
        image_request = {
            "S3Object": {
                "Bucket": plugin_params.input_folder_bucket,
                "Name": plugin_params.input_folder_root_path + image_path,
            }
        }
    else:
        with plugin_params.input_folder.get_download_stream(image_path) as stream:
            image_request = {"Bytes": stream.read()}
    response = plugin_params.api_client.detect_moderation_labels(Image=image_request, MinConfidence=minimum_score)
    return json.dumps(response)


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
