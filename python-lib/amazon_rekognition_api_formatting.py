# -*- coding: utf-8 -*-
"""Module with classes to format Amazon Rekognition API results
- extract meaningful columns from the API JSON response
- draw bounding boxes
"""

from typing import Dict, List
from enum import Enum

from PIL import Image
from fastcore.utils import store_attr

from api_image_formatting import ImageAPIResponseFormatterMeta
from plugin_io_utils import generate_unique, safe_json_loads
from image_utils import auto_rotate_image, draw_bounding_box_pil_image


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================


class UnsafeContentCategoryLevelEnum(Enum):
    TOP = "Top-level (simple)"
    SECOND = "Second-level (detailed)"


class UnsafeContentCategoryTopLevelEnum(Enum):
    EXPLICIT_NUDITY = "Explicit Nudity"
    SUGGESTIVE = "Suggestive"
    VIOLENCE = "Violence"
    VISUALLY_DISTURBING = "Visually Disturbing"


class UnsafeContentCategorySecondLevelEnum(Enum):
    NUDITY = "Nudity"
    GRAPHIC_MALE_NUDITY = "Graphic Male Nudity"
    GRAPHIC_FEMALE_NUDITY = "Graphic Female Nudity"
    SEXUAL_ACTIVITY = "Sexual Activity"
    ILLUSTRATED_NUDITY_OR_SEXUAL_ACTIVITY = "Illustrated Nudity Or Sexual Activity"
    ADULT_TOYS = "Adult Toys"
    FEMALE_SWIMWEAR_OR_UNDERWEAR = "Female Swimwear Or Underwear"
    MALE_SWIMWEAR_OR_UNDERWEAR = "Male Swimwear Or Underwear"
    PARTIAL_NUDITY = "Partial Nudity"
    REVEALING_CLOTHES = "Revealing Clothes"
    GRAPHIC_VIOLENCE_OR_GORE = "Graphic Violence Or Gore"
    PHYSICAL_VIOLENCE = "Physical Violence"
    WEAPON_VIOLENCE = "Weapon Violence"
    WEAPONS = "Weapons"
    SELF_INJURY = "Self Injury"
    EMACIATED_BODIES = "Emaciated Bodies"
    CORPSES = "Corpses"
    HANGING = "Hanging"


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class ObjectDetectionLabelingAPIResponseFormatter(ImageAPIResponseFormatterMeta):
    """
    Formatter class for Object Detection & Labeling API responses:
    - make sure response is valid JSON
    - extract object labels in a dataset
    - compute column descriptions
    - draw bounding boxes around objects with text containing label name and confidence score
    """

    def __init__(self, num_objects: int, orientation_correction: bool = True, **kwargs):
        store_attr()
        self._compute_column_description()

    def _compute_column_description(self):
        """Compute output column names and descriptions"""
        self.orientation_column = generate_unique("orientation_correction", self.input_df.keys(), self.column_prefix)
        self.label_list_column = generate_unique("label_list", self.input_df.keys(), self.column_prefix)
        self.label_name_columns = [
            generate_unique("label_" + str(n + 1) + "_name", self.input_df.keys(), self.column_prefix)
            for n in range(self.num_objects)
        ]
        self.label_score_columns = [
            generate_unique("label_" + str(n + 1) + "_score", self.input_df.keys(), self.column_prefix)
            for n in range(self.num_objects)
        ]
        self.column_description_dict[self.label_list_column] = "List of object labels from the API"
        self.column_description_dict[self.orientation_column] = "Orientation correction detected by the API"
        for n in range(self.num_objects):
            label_column = self.label_name_columns[n]
            score_column = self.label_score_columns[n]
            self.column_description_dict[label_column] = "Object label {} extracted by the API".format(n + 1)
            self.column_description_dict[score_column] = "Confidence score in label {} from 0 to 1".format(n + 1)

    def format_row(self, row: Dict) -> Dict:
        """Extract object and label lists from a row with an API response"""
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        row[self.label_list_column] = ""
        labels = sorted(response.get("Labels", []), key=lambda x: x.get("Confidence"), reverse=True)
        if len(labels) != 0:
            row[self.label_list_column] = [l.get("Name") for l in labels]
        for n in range(self.num_objects):
            if len(labels) > n:
                row[self.label_name_columns[n]] = labels[n].get("Name", "")
                row[self.label_score_columns[n]] = labels[n].get("Confidence", "")
            else:
                row[self.label_name_columns[n]] = ""
                row[self.label_score_columns[n]] = None
        if self.orientation_correction:
            row[self.orientation_column] = response.get("OrientationCorrection", "")
        return row

    def format_image(self, image: Image, response: Dict) -> Image:
        """Draw bounding boxes around detected objects"""
        bounding_box_list_dict = [
            {
                "name": label.get("Name", ""),
                "bbox_dict": instance.get("BoundingBox", {}),
                "confidence": float(instance.get("Confidence") / 100.0),
            }
            for label in response.get("Labels", [])
            for instance in label.get("Instances", [])
        ]
        if self.orientation_correction:
            detected_orientation = response.get("OrientationCorrection", "")
            (image, rotated) = auto_rotate_image(image, detected_orientation)
        bounding_box_list_dict = sorted(bounding_box_list_dict, key=lambda x: x.get("confidence", 0), reverse=True)
        for bounding_box_dict in bounding_box_list_dict:
            bbox_text = "{} - {:.1%} ".format(bounding_box_dict["name"], bounding_box_dict["confidence"])
            ymin = bounding_box_dict["bbox_dict"].get("Top")
            xmin = bounding_box_dict["bbox_dict"].get("Left")
            ymax = ymin + bounding_box_dict["bbox_dict"].get("Height")
            xmax = xmin + bounding_box_dict["bbox_dict"].get("Width")
            draw_bounding_box_pil_image(image, ymin, xmin, ymax, xmax, bbox_text)
        return image


class TextDetectionAPIResponseFormatter(ImageAPIResponseFormatterMeta):
    """
    Formatter class for Text Detection API responses:
    - make sure response is valid JSON
    - extract list of text transcriptions in a dataset
    - compute column descriptions
    - draw bounding boxes around detected text areas
    """

    def __init__(self, minimum_score: float = 0.0, orientation_correction: bool = True, **kwargs):
        store_attr()
        self._compute_column_description()

    def _compute_column_description(self):
        """Compute output column names and descriptions"""
        self.orientation_column = generate_unique("orientation_correction", self.input_df.keys(), self.column_prefix)
        self.text_column_list = generate_unique("detections_list", self.input_df.keys(), self.column_prefix)
        self.text_column_concat = generate_unique("detections_concat", self.input_df.keys(), self.column_prefix)
        self.column_description_dict[self.text_column_list] = "List of text detections from the API"
        self.column_description_dict[self.text_column_concat] = "Concatenated text detections from the API"
        self.column_description_dict[self.orientation_column] = "Orientation correction detected by the API"

    def format_row(self, row: Dict) -> Dict:
        """Extract detected text from a row with an API response"""
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        text_detections = response.get("TextDetections", [])
        text_detections_filtered = [
            t for t in text_detections if t.get("Confidence") >= self.minimum_score and t.get("ParentId") is None
        ]
        row[self.text_column_list] = ""
        row[self.text_column_concat] = ""
        if len(text_detections_filtered) != 0:
            row[self.text_column_list] = [t.get("DetectedText", "") for t in text_detections_filtered]
            row[self.text_column_concat] = " ".join(row[self.text_column_list])
        if self.orientation_correction:
            row[self.orientation_column] = response.get("OrientationCorrection", "")
        return row

    def format_image(self, image: Image, response: Dict) -> Image:
        """Draw bounding boxes around detected text"""
        text_detections = response.get("TextDetections", [])
        text_bounding_boxes = [
            t.get("Geometry", {}).get("BoundingBox", {})
            for t in text_detections
            if t.get("Confidence") >= self.minimum_score and t.get("ParentId") is None
        ]
        if self.orientation_correction:
            detected_orientation = response.get("OrientationCorrection", "")
            (image, rotated) = auto_rotate_image(image, detected_orientation)
        for bbox in text_bounding_boxes:
            ymin = bbox.get("Top")
            xmin = bbox.get("Left")
            ymax = bbox.get("Top") + bbox.get("Height")
            xmax = bbox.get("Left") + bbox.get("Width")
            draw_bounding_box_pil_image(image=image, ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)
        return image


class UnsafeContentAPIResponseFormatter(ImageAPIResponseFormatterMeta):
    """
    Formatter class for Unsafe Content API responses:
    - make sure response is valid JSON
    - extract moderation labels in a dataset
    - compute column descriptions
    """

    def __init__(
        self,
        category_level: UnsafeContentCategoryLevelEnum = UnsafeContentCategoryLevelEnum.TOP,
        content_categories_top_level: List[UnsafeContentCategoryTopLevelEnum] = [],
        content_categories_second_level: List[UnsafeContentCategorySecondLevelEnum] = [],
        **kwargs
    ):
        store_attr()
        if self.category_level == UnsafeContentCategoryLevelEnum.TOP:
            self.content_category_enum = UnsafeContentCategoryTopLevelEnum
            self.content_categories = content_categories_top_level
        else:
            self.content_category_enum = UnsafeContentCategorySecondLevelEnum
            self.content_categories = content_categories_second_level
        self._compute_column_description()

    def _compute_column_description(self):
        """Compute output column names and descriptions"""
        self.is_unsafe_column = generate_unique("unsafe_content", self.input_df.keys(), self.column_prefix)
        self.unsafe_list_column = generate_unique("unsafe_categories", self.input_df.keys(), self.column_prefix)
        self.column_description_dict[self.is_unsafe_column] = "Unsafe content detected by the API"
        self.column_description_dict[self.unsafe_list_column] = "List of unsafe content categories detected by the API"
        for n, m in self.content_category_enum.__members__.items():
            confidence_column = generate_unique(n.lower() + "_score", self.input_df.keys(), self.column_prefix)
            self.column_description_dict[confidence_column] = "Confidence score in category '{}' from 0 to 1".format(
                m.value
            )

    def format_row(self, row: Dict) -> Dict:
        """Extract moderation labels from a row with an API response"""
        raw_response = row[self.api_column_names.response]
        response = safe_json_loads(raw_response, self.error_handling)
        moderation_labels = response.get("ModerationLabels", [])
        row[self.is_unsafe_column] = False
        row[self.unsafe_list_column] = ""
        unsafe_list = []
        for category in self.content_categories:
            confidence_column = generate_unique(
                category.name.lower() + "_score", self.input_df.keys(), self.column_prefix
            )
            row[confidence_column] = ""
            if self.category_level == UnsafeContentCategoryLevelEnum.TOP:
                scores = [l.get("Confidence") for l in moderation_labels if l.get("ParentName", "") == category.value]
            else:
                scores = [l.get("Confidence") for l in moderation_labels if l.get("Name", "") == category.value]
            if len(scores) != 0:
                unsafe_list.append(str(category.value))
                row[confidence_column] = scores[0]
        if len(unsafe_list) != 0:
            row[self.is_unsafe_column] = True
            row[self.unsafe_list_column] = unsafe_list
        return row
