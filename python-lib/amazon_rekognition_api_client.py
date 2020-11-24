# -*- coding: utf-8 -*-
"""Module with utility functions to call the Amazon Rekognition API on a Dataiku managed folder"""

import logging
import json
from typing import AnyStr, Dict, Callable
from io import BytesIO

import dataiku
import boto3
from boto3.exceptions import Boto3Error
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image, UnidentifiedImageError
from ratelimit import limits, RateLimitException
from retry import retry
from fastcore.utils import store_attr

from plugin_io_utils import PATH_COLUMN
from image_utils import save_image_bytes, auto_rotate_image


class AmazonRekognitionAPIWrapper:
    """Wrapper class with helper methods to call the Amazon Rekognition API"""

    API_EXCEPTIONS = (Boto3Error, BotoCoreError, ClientError, UnidentifiedImageError)
    SUPPORTED_IMAGE_FORMATS = {"jpeg", "jpg", "png", "gif", "bmp", "webp", "ico"}
    RATELIMIT_EXCEPTIONS = (RateLimitException, OSError)
    RATELIMIT_RETRIES = 5

    def __init__(
        self,
        aws_access_key_id: AnyStr = None,
        aws_secret_access_key: AnyStr = None,
        aws_region_name: AnyStr = None,
        api_quota_period: int = 60,
        api_quota_rate_limit: int = 1800,
    ):
        store_attr()
        self.client = self.get_client()
        self.call_api_amazon_rekognition = self._build_call_api_amazon_rekognition()

    def get_client(self) -> boto3.client:
        """Initialize an Amazon Rekognition API client"""
        client = boto3.client(
            service_name="rekognition",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region_name,
        )
        logging.info("Credentials loaded")
        return client

    def _build_call_api_amazon_rekognition(self) -> Callable:
        """Build the API calling function for the Amazon Rekognition API with retrying and rate limiting"""

        @retry(exceptions=self.RATELIMIT_EXCEPTIONS, tries=self.RATELIMIT_RETRIES, delay=self.api_quota_period)
        @limits(calls=self.api_quota_rate_limit, period=self.api_quota_period)
        def call_api_amazon_rekognition(
            row: Dict,
            api_client_method_name: AnyStr,
            input_folder: dataiku.Folder,
            input_folder_is_s3: bool,
            input_folder_bucket: AnyStr,
            input_folder_root_path: AnyStr,
            orientation_correction: bool = False,
            num_objects: int = None,
            minimum_score: int = None,
            **kwargs
        ) -> AnyStr:
            """Call the Amazon Rekognition API with files stored in a Dataiku managed folder

            Used by `parallelizer.parallelizer` as `function` argument

            """
            image_path = row.get(PATH_COLUMN)
            pil_image = None
            if input_folder_is_s3:
                image_request = {
                    "S3Object": {"Bucket": input_folder_bucket, "Name": input_folder_root_path + image_path}
                }
            else:
                with input_folder.get_download_stream(image_path) as stream:
                    image_request = {"Bytes": stream.read()}
                    pil_image = Image.open(BytesIO(image_request["Bytes"]))
            if orientation_correction:
                # Need to use another API endpoint to retrieve the estimated orientation
                orientation_response = self.client.recognize_celebrities(Image=image_request)
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
            response = getattr(self.client, api_client_method_name)(**request_dict)
            if orientation_correction:
                response["OrientationCorrection"] = detected_orientation
            return json.dumps(response)

        return call_api_amazon_rekognition
