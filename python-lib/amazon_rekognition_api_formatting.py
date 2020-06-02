# -*- coding: utf-8 -*-
import os
import logging
import json
from typing import AnyStr, Dict, List
from enum import Enum

from PIL import Image, ImageFont, ImageDraw, UnidentifiedImageError
import pandas as pd

import dataiku

from plugin_io_utils import (
    API_COLUMN_NAMES_DESCRIPTION_DICT,
    IMAGE_PATH_COLUMN,
    ErrorHandlingEnum,
    build_unique_column_names,
    generate_unique,
    safe_json_loads,
    move_api_columns_to_end,
    upload_pil_image_to_folder,
)


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

BOUNDING_BOX_COLOR = "red"
BOUNDING_BOX_TEXT_FONT = ImageFont.truetype(
    font=os.path.join(dataiku.customrecipe.get_recipe_resource(), "SourceSansPro-Regular.otf"), size=64
)


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
    - draw bounding box
    """

    def __init__(
        self,
        input_df: pd.DataFrame,
        column_prefix: AnyStr = "api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        self.input_df = input_df
        self.output_df = None  # initialization before calling format_df
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
        self.output_df = df
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
        num_objects: int,
        input_folder: dataiku.Folder = None,
        column_prefix: AnyStr = "object_api",
        error_handling: ErrorHandlingEnum = ErrorHandlingEnum.LOG,
    ):
        super().__init__(input_df, column_prefix, error_handling)
        self.num_objects = num_objects
        self.input_folder = input_folder
        self.label_list_column = generate_unique("label_list", input_df.keys(), column_prefix)
        self.label_name_columns = [
            generate_unique("label_" + str(n + 1) + "_name", input_df.keys(), column_prefix) for n in range(num_objects)
        ]
        self.label_score_columns = [
            generate_unique("label_" + str(n + 1) + "_score", input_df.keys(), column_prefix)
            for n in range(num_objects)
        ]
        self._compute_column_description()

    def _compute_column_description(self):
        self.column_description_dict[self.label_list_column] = "List of object labels from the API"
        for n in range(self.num_objects):
            label_column = self.label_name_columns[n]
            score_column = self.label_score_columns[n]
            self.column_description_dict[label_column] = "Object label {} extracted by the API".format(n + 1)
            self.column_description_dict[score_column] = "Confidence score in label {} from 0 to 1".format(n + 1)

    def format_row(self, row: Dict) -> Dict:
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        labels = sorted(response.get("Labels", []), key=lambda x: x.get("Confidence"), reverse=True)
        row[self.label_list_column] = [l.get("Name") for l in labels]
        for n in range(self.num_objects):
            if len(labels) > n:
                row[self.label_name_columns[n]] = labels[n].get("Name", "")
                row[self.label_score_columns[n]] = labels[n].get("Confidence", "")
            else:
                row[self.label_name_columns[n]] = ""
                row[self.label_score_columns[n]] = None
        return row

    def draw_bounding_boxes(self, output_folder: dataiku.Folder):
        logging.info("Drawing bounding boxes and saving to output folder...")
        for _, row in self.output_df.iterrows():
            image_path = row[IMAGE_PATH_COLUMN]
            response = safe_json_loads(row[self.api_column_names.response])
            with self.input_folder.get_download_stream(image_path) as stream:
                try:
                    pil_image = Image.open(stream)
                    output_image = pil_image.convert(mode="RGB")
                    width, height = pil_image.size
                except (UnidentifiedImageError, OSError) as e:
                    logging.warning("Could not load image on path: " + image_path)
                    if self.error_handling == ErrorHandlingEnum.FAIL:
                        raise e
            if response != "" and len(response) != 0:
                draw = ImageDraw.Draw(output_image)
                for label in response.get("Labels", []):
                    for instance in label.get("Instances", []):
                        bbox = instance.get("BoundingBox", {})
                        left = width * bbox["Left"]
                        top = height * bbox["Top"]
                        w = width * bbox["Width"]
                        h = height * bbox["Height"]
                        margin = int(0.003 * max(width, height))  # reasonable design heuristic
                        draw.rectangle([left, top, left + w, top + h], outline=BOUNDING_BOX_COLOR, width=margin)
                        bbox_text = "{} - {:.1%}".format(label.get("Name", ""), instance.get("Confidence") / 100.0)
                        # TODO adjust text size to scale
                        draw.text(
                            [left + 2 * margin, top + 2 * margin],  # reasonable design heuristic
                            text=bbox_text,
                            fill=BOUNDING_BOX_COLOR,
                            font=BOUNDING_BOX_TEXT_FONT,
                        )
                # TODO use same extension as original file
                upload_pil_image_to_folder(output_image, output_folder, image_path)
        logging.info("Drawing bounding boxes and saving to output folder: Done.")


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
