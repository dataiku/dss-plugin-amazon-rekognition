# -*- coding: utf-8 -*-
"""Module with utility classes for validating and loading plugin parameters"""

import logging
from typing import Dict, AnyStr, List

import dataiku
import boto3
import pandas as pd
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role

from amazon_rekognition_api_client import get_client
from plugin_io_utils import IMAGE_PATH_COLUMN, ErrorHandlingEnum
from dku_io_utils import generate_path_list
from amazon_rekognition_api_formatting import (
    UnsafeContentCategoryLevelEnum,
    UnsafeContentCategoryTopLevelEnum,
    UnsafeContentCategorySecondLevelEnum,
)


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class PluginParamValidationError(ValueError):
    """Custom exception raised when the the plugin parameters chosen by the user are invalid"""

    pass


class PluginParams:
    """Class to hold plugin parameters"""

    def __init__(
        self,
        api_client: boto3.client,
        input_folder: dataiku.Folder,
        input_df: pd.DataFrame,
        output_dataset: dataiku.Dataset,
        api_quota_rate_limit: int,
        api_quota_period: int,
        parallel_workers: int,
        minimum_score: float,
        error_handling: ErrorHandlingEnum,
        num_objects: int = 10,
        orientation_correction: bool = False,
        output_folder: dataiku.Folder = None,
        input_folder_is_s3: bool = False,
        input_folder_bucket: AnyStr = "",
        input_folder_root_path: AnyStr = "",
        unsafe_content_category_level: UnsafeContentCategoryLevelEnum = UnsafeContentCategoryLevelEnum.TOP,
        unsafe_content_categories_top_level: List[UnsafeContentCategoryTopLevelEnum] = [],
        unsafe_content_categories_second_level: List[UnsafeContentCategorySecondLevelEnum] = [],
    ):
        self.api_client = api_client
        self.input_folder = input_folder
        self.input_df = input_df
        self.input_folder_is_s3 = input_folder_is_s3
        self.input_folder_bucket = input_folder_bucket
        self.input_folder_root_path = input_folder_root_path
        self.output_dataset = output_dataset
        self.output_folder = output_folder
        self.api_quota_rate_limit = api_quota_rate_limit
        self.api_quota_period = api_quota_period
        self.parallel_workers = parallel_workers
        self.num_objects = num_objects
        self.minimum_score = minimum_score
        self.orientation_correction = orientation_correction
        self.error_handling = error_handling
        self.unsafe_content_category_level = unsafe_content_category_level
        self.unsafe_content_categories_top_level = unsafe_content_categories_top_level
        self.unsafe_content_categories_second_level = unsafe_content_categories_second_level


class PluginParamsLoader:
    """Class to validate and load plugin parameters"""

    def validate_input_params(self) -> Dict:
        """Validate input parameters"""
        input_params_dict = {}
        input_folder_names = get_input_names_for_role("input_folder")
        if len(input_folder_names) == 0:
            raise PluginParamValidationError("Please specify input folder")
        input_params_dict["input_folder"] = dataiku.Folder(input_folder_names[0])
        image_path_list = [
            p
            for p in generate_path_list(input_params_dict["input_folder"])
            if p.split(".")[-1].lower() in {"jpeg", "jpg", "png"}
        ]
        if len(image_path_list) == 0:
            raise PluginParamValidationError("No images of supported format (PNG or JPG) were found in input folder")
        input_params_dict["input_df"] = pd.DataFrame(image_path_list, columns=[IMAGE_PATH_COLUMN])
        input_params_dict["input_folder_is_s3"] = input_params_dict["input_folder"].get_info().get("type", "") == "S3"
        if input_params_dict["input_folder_is_s3"]:
            input_folder_access_info = input_params_dict["input_folder"].get_info().get("accessInfo", {})
            input_params_dict["input_folder_bucket"] = input_folder_access_info.get("bucket")
            input_params_dict["input_folder_root_path"] = str(input_folder_access_info.get("root", ""))[1:]
            logging.info(
                "Input folder is on Amazon S3 with bucket: {} and root path: {}".format(
                    input_params_dict["input_folder_bucket"], input_params_dict["input_folder_root_path"]
                )
            )
        return input_params_dict

    def validate_output_params(self) -> Dict:
        """Validate output parameters"""
        output_params_dict = {}
        # Mandatory output dataset
        output_dataset_names = get_output_names_for_role("output_dataset")  # mandatory output
        if len(output_dataset_names) == 0:
            raise PluginParamValidationError("Please specify output folder")
        output_params_dict["output_dataset"] = dataiku.Dataset(output_dataset_names[0])
        # Mandatory output folder
        output_folder_names = get_output_names_for_role("output_folder")  # optional output
        output_params_dict["output_folder"] = None
        if len(output_folder_names) != 0:
            output_params_dict["output_folder"] = dataiku.Folder(output_folder_names[0])
        return output_params_dict

    def validate_recipe_params(self) -> Dict:
        recipe_params_dict = {}
        recipe_config = get_recipe_config()
        recipe_params_dict["num_objects"] = int(recipe_config.get("num_objects", 1))
        if recipe_params_dict["num_objects"] < 1:
            raise PluginParamValidationError("Number of objects must be greater than 1")
        recipe_params_dict["minimum_score"] = int(recipe_config.get("minimum_score", 0) * 100)
        if recipe_params_dict["minimum_score"] < 0 or recipe_params_dict["minimum_score"] > 100:
            raise PluginParamValidationError("Minimum confidence score must be between 0 and 1")
        recipe_params_dict["orientation_correction"] = bool(recipe_config.get("orientation_correction", False))
        recipe_params_dict["error_handling"] = ErrorHandlingEnum[recipe_config.get("error_handling")]
        if "category_level" in recipe_config:
            recipe_params_dict["unsafe_content_category_level"] = UnsafeContentCategoryLevelEnum[
                recipe_config.get("category_level")
            ]
            recipe_params_dict["unsafe_content_categories_top_level"] = [
                UnsafeContentCategoryTopLevelEnum[i] for i in recipe_config.get("content_categories_top_level", [])
            ]
            recipe_params_dict["unsafe_content_categories_second_level"] = [
                UnsafeContentCategorySecondLevelEnum[i]
                for i in recipe_config.get("content_categories_second_level", [])
            ]
            if (
                len(recipe_params_dict["unsafe_content_categories_top_level"]) == 0
                or len(recipe_params_dict["unsafe_content_categories_second_level"]) == 0
            ):
                raise PluginParamValidationError("Choose at least one category")
        logging.info("Validated plugin recipe parameters: {}".format(recipe_params_dict))
        return recipe_params_dict

    def validate_preset_params(self) -> Dict:
        """Validate API configuration preset parameters"""
        preset_params_dict = {}
        recipe_config = get_recipe_config()
        api_configuration_preset = recipe_config.get("api_configuration_preset", {})
        preset_params_dict["api_quota_period"] = int(api_configuration_preset.get("api_quota_period", 1))
        if preset_params_dict["api_quota_period"] < 1:
            raise PluginParamValidationError("API quota period must be greater than 1")
        preset_params_dict["api_quota_rate_limit"] = int(api_configuration_preset.get("api_quota_rate_limit", 1))
        if preset_params_dict["api_quota_rate_limit"] < 1:
            raise PluginParamValidationError("API quota rate limit must be greater than 1")
        preset_params_dict["parallel_workers"] = int(api_configuration_preset.get("parallel_workers", 1))
        if preset_params_dict["parallel_workers"] < 1 or preset_params_dict["parallel_workers"] > 100:
            raise PluginParamValidationError("Concurrency must be between 1 and 100")
        logging.info("Validated preset parameters: {}".format(preset_params_dict))
        preset_params_dict["api_client"] = get_client(
            aws_access_key_id=api_configuration_preset.get("aws_access_key_id"),
            aws_secret_access_key=api_configuration_preset.get("aws_secret_access_key"),
            aws_region_name=api_configuration_preset.get("aws_region_name"),
        )
        return preset_params_dict

    def load_validate_params(self) -> PluginParams:
        input_params_dict = self.validate_input_params()
        output_params_dict = self.validate_output_params()
        recipe_params_dict = self.validate_recipe_params()
        preset_params_dict = self.validate_preset_params()
        plugin_params = PluginParams(
            **input_params_dict, **output_params_dict, **recipe_params_dict, **preset_params_dict
        )
        return plugin_params
