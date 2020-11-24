# -*- coding: utf-8 -*-
"""Unsafe Content Moderation recipe script"""

from plugin_params_loader import PluginParamsLoader, RecipeID
from parallelizer import parallelizer
from amazon_rekognition_api_formatting import UnsafeContentAPIResponseFormatter
from dku_io_utils import set_column_description

params = PluginParamsLoader(RecipeID.UNSAFE_CONTENT_MODERATION).validate_load_params()

df = parallelizer(
    function=params.api_wrapper.call_api_amazon_rekognition,
    api_client_method_name="detect_moderation_labels",
    exceptions=params.api_wrapper.API_EXCEPTIONS,
    **vars(params)
)

api_formatter = UnsafeContentAPIResponseFormatter(**vars(params))
output_df = api_formatter.format_df(df)
params.output_dataset.write_with_schema(output_df)
set_column_description(params.output_dataset, api_formatter.column_description_dict)
