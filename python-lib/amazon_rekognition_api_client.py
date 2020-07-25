# -*- coding: utf-8 -*-
import logging
from typing import AnyStr
import boto3
from boto3.exceptions import Boto3Error
from botocore.exceptions import BotoCoreError, ClientError
from PIL import UnidentifiedImageError

# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = (Boto3Error, BotoCoreError, ClientError, UnidentifiedImageError)
SUPPORTED_IMAGE_FORMATS = ["jpeg", "jpg", "png"]

# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def get_client(api_configuration_preset):
    client = boto3.client(
        service_name="rekognition",
        aws_access_key_id=api_configuration_preset.get("aws_access_key"),
        aws_secret_access_key=api_configuration_preset.get("aws_secret_key"),
        region_name=api_configuration_preset.get("aws_region"),
    )
    logging.info("Credentials loaded")
    return client


def supported_image_format(filepath: AnyStr):
    extension = filepath.split(".")[-1].lower()
    return extension in SUPPORTED_IMAGE_FORMATS
