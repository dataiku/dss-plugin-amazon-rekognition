# -*- coding: utf-8 -*-
"""Module with utility classes for validating and loading plugin parameters"""

import logging
from typing import Dict, AnyStr, List
from enum import Enum

import pandas as pd
from fastcore.utils import store_attr

import dataiku
from dataiku.customrecipe import get_recipe_config, get_input_names_for_role, get_output_names_for_role

from amazon_rekognition_api_client import AmazonRekognitionAPIWrapper
from plugin_io_utils import PATH_COLUMN, ErrorHandling
from dku_io_utils import generate_path_df
from amazon_rekognition_api_formatting import (
    UnsafeContentCategoryLevelEnum,
    UnsafeContentCategoryTopLevelEnum,
    UnsafeContentCategorySecondLevelEnum,
)


class RecipeID(Enum):
    """Enum class to identify each recipe"""

    OBJECT_DETECTION_LABELING = "object_api"
    TEXT_DETECTION = "text_api"
    UNSAFE_CONTENT_MODERATION = "moderation_api"


class PluginParamValidationError(ValueError):
    """Custom exception raised when the the plugin parameters chosen by the user are invalid"""

    pass


class PluginParams:
    """Class to hold plugin parameters"""

    def __init__(
        self,
        api_wrapper: AmazonRekognitionAPIWrapper,
        input_folder: dataiku.Folder,
        input_df: pd.DataFrame,
        column_prefix: AnyStr,
        output_dataset: dataiku.Dataset,
        api_quota_rate_limit: int,
        api_quota_period: int,
        parallel_workers: int,
        minimum_score: float,
        error_handling: ErrorHandling,
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
        store_attr()


class PluginParamsLoader:
    """Class to validate and load plugin parameters"""

    DOC_URL = "https://www.dataiku.com/product/plugins/amazon-rekognition/"

    def __init__(self, recipe_id: RecipeID):
        self.recipe_id = recipe_id
        self.column_prefix = self.recipe_id.value
        self.recipe_config = get_recipe_config()

    def validate_input_params(self) -> Dict:
        """Validate input parameters"""
        input_params = {}
        input_folder_names = get_input_names_for_role("input_folder")
        if len(input_folder_names) == 0:
            raise PluginParamValidationError("Please specify input folder")
        input_params["input_folder"] = dataiku.Folder(input_folder_names[0])
        input_params["input_df"] = generate_path_df(
            folder=input_params["input_folder"],
            file_extensions=AmazonRekognitionAPIWrapper.SUPPORTED_IMAGE_FORMATS,
            path_column=PATH_COLUMN,
        )
        input_params["input_folder_is_s3"] = input_params["input_folder"].get_info().get("type", "") == "S3"
        if input_params["input_folder_is_s3"]:
            input_folder_access_info = input_params["input_folder"].get_info().get("accessInfo", {})
            input_params["input_folder_bucket"] = input_folder_access_info.get("bucket")
            input_params["input_folder_root_path"] = str(input_folder_access_info.get("root", ""))[1:]
            logging.info(
                f"Input folder is on Amazon S3 with bucket: {input_params['input_folder_bucket']} "
                + f"and root path: {input_params['input_folder_root_path']}"
            )
        return input_params

    def validate_output_params(self) -> Dict:
        """Validate output parameters"""
        output_params = {}
        # Mandatory output dataset
        output_dataset_names = get_output_names_for_role("output_dataset")
        if len(output_dataset_names) == 0:
            raise PluginParamValidationError("Please specify output dataset")
        output_params["output_dataset"] = dataiku.Dataset(output_dataset_names[0])
        # Optional output folder
        output_folder_names = get_output_names_for_role("output_folder")
        output_params["output_folder"] = None
        if self.recipe_id != RecipeID.UNSAFE_CONTENT_MODERATION:
            if len(output_folder_names) == 0:
                raise PluginParamValidationError("Please specify output folder")
            else:
                output_params["output_folder"] = dataiku.Folder(output_folder_names[0])
        return output_params

    def validate_recipe_params(self) -> Dict:
        """Validate recipe parameters for all recipes"""
        recipe_params = {}
        # Applies to several recipes
        minimum_score = self.recipe_config.get("minimum_score")
        if not isinstance(minimum_score, (int, float)):
            raise PluginParamValidationError(f"Invalid minimum score parameter: {minimum_score}")
        recipe_params["minimum_score"] = int(minimum_score * 100)
        if recipe_params["minimum_score"] < 0 or recipe_params["minimum_score"] > 100:
            raise PluginParamValidationError("Minimum confidence score must be between 0 and 1")
        recipe_params["error_handling"] = ErrorHandling[self.recipe_config.get("error_handling")]
        recipe_params["orientation_correction"] = bool(self.recipe_config.get("orientation_correction", False))
        # Applies to object detection & labeling
        if self.recipe_id == RecipeID.OBJECT_DETECTION_LABELING:
            num_objects = self.recipe_config.get("num_objects")
            if not isinstance(num_objects, (int, float)):
                raise PluginParamValidationError(f"Invalid number of labels parameter: {num_objects}")
            if num_objects < 1:
                raise PluginParamValidationError("Number of labels must be greater than 1")
            recipe_params["num_objects"] = num_objects
        # Applies to unsafe content moderation
        if self.recipe_id == RecipeID.UNSAFE_CONTENT_MODERATION:
            recipe_params["unsafe_content_category_level"] = UnsafeContentCategoryLevelEnum[
                self.recipe_config.get("category_level")
            ]
            recipe_params["unsafe_content_categories_top_level"] = [
                UnsafeContentCategoryTopLevelEnum[i] for i in self.recipe_config.get("content_categories_top_level", [])
            ]
            recipe_params["unsafe_content_categories_second_level"] = [
                UnsafeContentCategorySecondLevelEnum[i]
                for i in self.recipe_config.get("content_categories_second_level", [])
            ]
            if (
                len(recipe_params["unsafe_content_categories_top_level"]) == 0
                or len(recipe_params["unsafe_content_categories_second_level"]) == 0
            ):
                raise PluginParamValidationError("Choose at least one unsafe content category")
        logging.info(f"Validated plugin recipe parameters: {recipe_params}")
        return recipe_params

    def validate_preset_params(self) -> Dict:
        """Validate API configuration preset parameters"""
        preset_params = {}
        api_configuration_preset = self.recipe_config.get("api_configuration_preset", {})
        if not api_configuration_preset:
            raise PluginParamValidationError(f"Please specify an API configuration preset according to {self.DOC_URL}")
        preset_params["api_quota_period"] = int(api_configuration_preset.get("api_quota_period", 1))
        if preset_params["api_quota_period"] < 1:
            raise PluginParamValidationError("API quota period must be greater than 1")
        preset_params["api_quota_rate_limit"] = int(api_configuration_preset.get("api_quota_rate_limit", 1))
        if preset_params["api_quota_rate_limit"] < 1:
            raise PluginParamValidationError("API quota rate limit must be greater than 1")
        preset_params["parallel_workers"] = int(api_configuration_preset.get("parallel_workers", 1))
        if preset_params["parallel_workers"] < 1 or preset_params["parallel_workers"] > 100:
            raise PluginParamValidationError("Concurrency must be between 1 and 100")
        logging.info(f"Validated preset parameters: {preset_params}")
        preset_params["api_wrapper"] = AmazonRekognitionAPIWrapper(
            aws_access_key_id=api_configuration_preset.get("aws_access_key_id"),
            aws_secret_access_key=api_configuration_preset.get("aws_secret_access_key"),
            aws_region_name=api_configuration_preset.get("aws_region_name"),
            api_quota_period=preset_params["api_quota_period"],
            api_quota_rate_limit=preset_params["api_quota_rate_limit"],
        )
        return preset_params

    def validate_load_params(self) -> PluginParams:
        """Validate and load all parameters into a `PluginParams` instance"""
        input_params = self.validate_input_params()
        output_params = self.validate_output_params()
        recipe_params = self.validate_recipe_params()
        preset_params = self.validate_preset_params()
        plugin_params = PluginParams(
            column_prefix=self.column_prefix, **input_params, **output_params, **recipe_params, **preset_params
        )
        return plugin_params
