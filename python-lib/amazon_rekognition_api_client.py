# -*- coding: utf-8 -*-
"""Module with a wrapper class around the Amazon Rekognition API"""

import logging
import json
from typing import AnyStr, Dict
from io import BytesIO

import boto3
from boto3.exceptions import Boto3Error
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image, UnidentifiedImageError

import dataiku

from plugin_io_utils import IMAGE_PATH_COLUMN
from plugin_image_utils import save_image_bytes, auto_rotate_image

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = (Boto3Error, BotoCoreError, ClientError, UnidentifiedImageError)


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def get_client(aws_access_key_id: AnyStr, aws_secret_access_key: AnyStr, aws_region_name: AnyStr) -> boto3.client:
    client = boto3.client(
        service_name="rekognition",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region_name,
    )
    logging.info("Credentials loaded")
    return client


def call_api_generic(
    row: Dict,
    api_client: boto3.client,
    api_client_method_name: AnyStr,
    input_folder: dataiku.Folder,
    input_folder_is_s3: bool,
    input_folder_bucket: AnyStr,
    input_folder_root_path: AnyStr,
    orientation_correction: bool = False,
    num_objects: int = None,
    minimum_score: int = None,
) -> AnyStr:
    image_path = row.get(IMAGE_PATH_COLUMN)
    pil_image = None
    if input_folder_is_s3:
        image_request = {"S3Object": {"Bucket": input_folder_bucket, "Name": input_folder_root_path + image_path}}
    else:
        with input_folder.get_download_stream(image_path) as stream:
            image_request = {"Bytes": stream.read()}
            pil_image = Image.open(BytesIO(image_request["Bytes"]))
    if orientation_correction:
        # Need to use another API endpoint to retrieve the estimated orientation
        orientation_response = api_client.recognize_celebrities(Image=image_request)
        detected_orientation = orientation_response.get("OrientationCorrection", "")
        if pil_image is None:
            with input_folder.get_download_stream(image_path) as stream:
                pil_image = Image.open(stream)
        (rotated_image, rotated) = auto_rotate_image(pil_image, detected_orientation)
        if rotated:
            logging.info("Corrected image orientation: {}".format(image_path))
            image_request = {"Bytes": save_image_bytes(rotated_image, image_path).getvalue()}
    request_dict = {"Image": image_request}
    if num_objects:
        request_dict["MaxLabels"] = num_objects
    if minimum_score:
        request_dict["MinConfidence"] = minimum_score
    response = getattr(api_client, api_client_method_name)(**request_dict)
    if orientation_correction:
        response["OrientationCorrection"] = detected_orientation
    return json.dumps(response)
