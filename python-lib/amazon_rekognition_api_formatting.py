# -*- coding: utf-8 -*-
import logging
import json
from typing import AnyStr, Dict, List
from enum import Enum

from PIL import ImageDraw
import pandas as pd

from plugin_io_utils import (
    API_COLUMN_NAMES_DESCRIPTION_DICT,
    ErrorHandlingEnum,
    build_unique_column_names,
    generate_unique,
    safe_json_loads,
    move_api_columns_to_end,
)


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================


class EntityTypeEnum(Enum):
    COMMERCIAL_ITEM = "Commercial item"
    DATE = "Date"
    EVENT = "Event"
    LOCATION = "Location"
    ORGANIZATION = "Organization"
    OTHER = "Other"
    PERSON = "Person"
    QUANTITY = "Quantity"
    TITLE = "Title"


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class GenericAPIFormatter:
    """
    Geric Formatter class for API responses:
    - initialize with generic parameters
    - compute generic column descriptions
    - apply format_row to dataframe
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        column_prefix: AnyStr = "api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        self.input_df = input_df
        self.column_prefix = column_prefix
        self.error_handling = error_handling
        self.api_column_names = build_unique_column_names(input_df, column_prefix)
        self.column_description_dict = {
            v: API_COLUMN_NAMES_DESCRIPTION_DICT[k] for k, v in self.api_column_names._asdict().items()
        }

    def format_row(self, row: Dict) -> Dict:
        return row

    def format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Formatting API results...")
        df = df.apply(func=self.format_row, axis=1)
        df = move_api_columns_to_end(df, self.api_column_names, self.error_handling)
        logging.info("Formatting API results: Done.")
        return df


class ObjectDetectionLabelingAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for Object Detection & Labeling API responses:
    - make sure response is valid JSON
    - extract object labels in a dataset
    - output an image with bounding boxes for each object
    - compute column descriptions
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        column_prefix: AnyStr = "lang_detect_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(input_df, column_prefix, error_handling)
        self.language_code_column = generate_unique("language_code", input_df.keys(), self.column_prefix)
        self.language_score_column = generate_unique("language_score", input_df.keys(), self.column_prefix)
        self._compute_column_description()

    def _compute_column_description(self):
        self.column_description_dict[self.language_code_column] = "Language code from the API in ISO 639 format"
        self.column_description_dict[self.language_score_column] = "Confidence score of the API from 0 to 1"

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        row[self.language_code_column] = ""
        row[self.language_score_column] = None
        languages = response.get("Languages", [])
        if len(languages) != 0:
            row[self.language_code_column] = languages[0].get("LanguageCode", "")
            row[self.language_score_column] = languages[0].get("Score", None)
        return row


class TextDetectionAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for Text Detection API responses:
    - make sure response is valid JSON
    - extract text in a dataset
    - output an image with bounding boxes for each text
    - compute column descriptions
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        column_prefix: AnyStr = "sentiment_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(input_df, column_prefix, error_handling)
        self.sentiment_prediction_column = generate_unique("prediction", input_df.keys(), column_prefix)
        self.sentiment_score_column_dict = {
            p: generate_unique("score_" + p.lower(), input_df.keys(), column_prefix)
            for p in ["Positive", "Neutral", "Negative", "Mixed"]
        }
        self._compute_column_description()

    def _compute_column_description(self):
        self.column_description_dict[
            self.sentiment_prediction_column
        ] = "Sentiment prediction from the API (POSITIVE/NEUTRAL/NEGATIVE/MIXED)"
        for prediction, column_name in self.sentiment_score_column_dict.items():
            self.column_description_dict[column_name] = "Confidence score in the {} prediction from 0 to 1".format(
                prediction.upper()
            )

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        row[self.sentiment_prediction_column] = response.get("Sentiment", "")
        sentiment_score = response.get("SentimentScore", {})
        for prediction, column_name in self.sentiment_score_column_dict.items():
            row[column_name] = None
            score = sentiment_score.get(prediction)
            if score is not None:
                row[column_name] = round(score, 3)
        return row


class UnsafeContentAPIFormatter(GenericAPIFormatter):
    """
    Formatter class for Unsafe Content API responses:
    - make sure response is valid JSON
    - extract moderation labels in a dataset
    - compute column descriptions
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        entity_types: List,
        minimum_score: float,
        column_prefix: AnyStr = "entity_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(input_df, column_prefix, error_handling)
        self.entity_types = entity_types
        self.minimum_score = float(minimum_score)
        self._compute_column_description()

    def _compute_column_description(self):
        for n, m in EntityTypeEnum.__members__.items():
            entity_type_column = generate_unique("entity_type_" + n.lower(), self.input_df.keys(), self.column_prefix)
            self.column_description_dict[entity_type_column] = "List of '{}' entities recognized by the API".format(
                str(m.value)
            )

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        entities = response.get("Entities", [])
        selected_entity_types = sorted([e.name for e in self.entity_types])
        for n in selected_entity_types:
            entity_type_column = generate_unique("entity_type_" + n.lower(), row.keys(), self.column_prefix)
            row[entity_type_column] = [
                e.get("Text")
                for e in entities
                if e.get("Type", "") == n and float(e.get("Score", 0)) >= self.minimum_score
            ]
            if len(row[entity_type_column]) == 0:
                row[entity_type_column] = ""
        return row


def detect_labels(image_file, client):
    row = {}
    response = client.detect_labels(Image={"Bytes": image_file.read()})
    labels = [l["Name"] for l in response.get("Labels")]
    if len(labels):
        row["detected_labels"] = json.dumps(labels)
    return row, response, labels


def detect_objects(image_file, client):
    row = {}
    response = client.detect_labels(Image={"Bytes": image_file.read()})
    bbox_list = _format_bounded_boxes(response.get("Labels"))
    if len(bbox_list):
        row["detected_objects"] = json.dumps(bbox_list)
    return row, response, bbox_list


def detect_adult_content(image_file, client):
    row = {"adult_score": 0, "suggestive_score": 0, "violence_score": 0}
    response = client.detect_moderation_labels(Image={"Bytes": image_file.read()})

    for m in response.get("ModerationLabels", []):
        if m["Name"] == "Explicit Nudity" or m["ParentName"] == "Explicit Nudity":
            row["adult_score"] = max(row["adult_score"], m["Confidence"])
        if m["Name"] == "Suggestive" or m["ParentName"] == "Suggestive":
            row["suggestive_score"] = max(row["suggestive_score"], m["Confidence"])
        if m["Name"] == "Violence" or m["ParentName"] == "Violence":
            row["violence_score"] = max(row["violence_score"], m["Confidence"])
    row["is_adult_content"] = row["adult_score"] > 0.5
    row["is_suggestive_content"] = row["suggestive_score"] > 0.5
    row["is_violent_content"] = row["suggestive_score"] > 0.5
    return row, response


def _format_bounded_boxes(raw_bbox):
    ret = []
    for label in raw_bbox:
        for instance in label["Instances"]:
            ret.append(
                {
                    "label": label["Name"],
                    "score": instance["Confidence"],
                    "top": instance["BoundingBox"]["Top"],
                    "left": instance["BoundingBox"]["Left"],
                    "width": instance["BoundingBox"]["Width"],
                    "height": instance["BoundingBox"]["Height"],
                }
            )
    return ret


"""
bbox_list: list of {label, score, top, left, width, height}
  * score is supposed to be in 0-1 range
  * top, left, width, height are normalized (0-1 range)
"""


def draw_bounding_boxes(pil_image, bbox_list):
    output_image = pil_image.convert(mode="RGB")
    width, height = pil_image.size
    draw = ImageDraw.Draw(output_image)

    for bbox in bbox_list:
        left = width * bbox["left"]
        top = height * bbox["top"]
        w = width * bbox["width"]
        h = height * bbox["height"]
        draw.rectangle([left, top, left + w, top + h], outline="#00FF00")
        draw.text([left, top], bbox["label"])  # add proba and improve font

    return output_image