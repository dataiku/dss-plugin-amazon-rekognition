# -*- coding: utf-8 -*-
import json
import logging
from typing import Dict, AnyStr
from ratelimit import limits, RateLimitException
from retry import retry
from PIL import Image
from io import BytesIO


from plugin_params_loader import PluginParamsLoader
from amazon_rekognition_api_client import API_EXCEPTIONS
from plugin_io_utils import IMAGE_PATH_COLUMN
from dku_io_utils import set_column_description
from plugin_image_utils import save_image_bytes, auto_rotate_image
from api_parallelizer import api_parallelizer
from amazon_rekognition_api_formatting import ObjectDetectionLabelingAPIFormatter


# ==============================================================================
# SETUP
# ==============================================================================

plugin_params = PluginParamsLoader().load_validate_params()
column_prefix = "object_api"


# ==============================================================================
# RUN
# ==============================================================================


@retry((RateLimitException, OSError), delay=plugin_params.api_quota_period, tries=5)
@limits(calls=plugin_params.api_quota_rate_limit, period=plugin_params.api_quota_period)
def call_api_object_detection(row: Dict, num_objects: int, minimum_score: int, orientation_correction: bool) -> AnyStr:
    image_path = row.get(IMAGE_PATH_COLUMN)
    pil_image = None
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
            pil_image = Image.open(BytesIO(image_request["Bytes"]))
    if orientation_correction:
        # Need to use another API endpoint to retrieve the estimated orientation
        orientation_response = plugin_params.api_client.recognize_celebrities(Image=image_request)
        detected_orientation = orientation_response.get("OrientationCorrection", "")
        if pil_image is None:
            with plugin_params.input_folder.get_download_stream(image_path) as stream:
                pil_image = Image.open(stream)
        (rotated_image, rotated) = auto_rotate_image(pil_image, detected_orientation)
        if rotated:
            logging.info("Corrected image orientation: {}".format(image_path))
            image_request = {"Bytes": save_image_bytes(rotated_image, image_path).getvalue()}
    response = plugin_params.api_client.detect_labels(
        Image=image_request, MaxLabels=num_objects, MinConfidence=minimum_score
    )
    if orientation_correction:
        response["OrientationCorrection"] = detected_orientation
    return json.dumps(response)


df = api_parallelizer(
    input_df=plugin_params.input_df,
    api_call_function=call_api_object_detection,
    api_exceptions=API_EXCEPTIONS,
    parallel_workers=plugin_params.parallel_workers,
    error_handling=plugin_params.error_handling,
    num_objects=plugin_params.num_objects,
    minimum_score=plugin_params.minimum_score,
    orientation_correction=plugin_params.orientation_correction,
    column_prefix=column_prefix,
)

api_formatter = ObjectDetectionLabelingAPIFormatter(
    input_df=plugin_params.input_df,
    num_objects=plugin_params.num_objects,
    orientation_correction=plugin_params.orientation_correction,
    input_folder=plugin_params.input_folder,
    error_handling=plugin_params.error_handling,
    parallel_workers=plugin_params.parallel_workers,
    column_prefix=column_prefix,
)
output_df = api_formatter.format_df(df)

plugin_params.output_dataset.write_with_schema(output_df)
set_column_description(
    output_dataset=plugin_params.output_dataset, column_description_dict=api_formatter.column_description_dict
)
if plugin_params.output_folder is not None:
    api_formatter.format_save_images(plugin_params.output_folder)
