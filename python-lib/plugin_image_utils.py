# -*- coding: utf-8 -*-
import os
from typing import List, AnyStr

import numpy as np
from dataiku.customrecipe import get_recipe_resource
from PIL import Image, ImageFont, ImageDraw


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

BOUNDING_BOX_COLOR = "red"
BOUNDING_BOX_FONT_PATH = os.path.join(get_recipe_resource(), "SourceSansPro-Regular.ttf")
BOUNDING_BOX_FONT_DEFAULT_SIZE = 18


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def scale_bounding_box_font(image: Image, text_line_list: List[AnyStr], bounding_box_width: int) -> ImageFont:
    """
    Compute font for bounding box text to enforce specific design guidelines for short text:
    - use custom font from BOUNDING_BOX_FONT_PATH (bundled in the resource folder of the plugin)
    - scale font size to fit the text to a percentage of the bounding box width
        * 120% - narrow bounding boxes below 15% of image width
        * 120% - narrow bounding boxes below 15% of image width
        * 80% - default
        * 33% - large bounding boxes above 33% of image width
    - bucket font size in increments (2, 4, 6, 8, ...) to homogenize font sizing
    """
    # Initialize font
    im_width, im_height = image.size
    font_default_size = ImageFont.truetype(font=BOUNDING_BOX_FONT_PATH, size=BOUNDING_BOX_FONT_DEFAULT_SIZE)
    text_width_default_size = max([font_default_size.getsize(t)[0] for t in text_line_list])
    # Start with a default font size at 80% of bounding box width
    scaled_font_size = int(0.8 * bounding_box_width / text_width_default_size * BOUNDING_BOX_FONT_DEFAULT_SIZE)
    scaled_font = font_default_size.font_variant(size=scaled_font_size)
    text_width_scaled_size = max([scaled_font.getsize(t)[0] for t in text_line_list])
    # Narrow bounding boxes below 15% of image width
    if text_width_scaled_size < np.ceil(0.15 * im_width):
        scaled_font_size = int(1.2 * bounding_box_width / text_width_default_size * BOUNDING_BOX_FONT_DEFAULT_SIZE)
    # Large bounding boxes above 33% of image width
    elif text_width_scaled_size > np.ceil(0.33 * im_width):
        scaled_font_size = int(0.33 * bounding_box_width / text_width_default_size * BOUNDING_BOX_FONT_DEFAULT_SIZE)
    # Bucket font size in increments (2, 4, 6, 8, ...) to homogenize font sizing
    scaled_font_size = 2 * int(np.ceil(scaled_font_size / 2.0))
    scaled_font = font_default_size.font_variant(size=scaled_font_size)
    return scaled_font


def draw_bounding_box_pil_image(
    image: Image,
    ymin: float,
    xmin: float,
    ymax: float,
    xmax: float,
    text: AnyStr = "",
    color: AnyStr = BOUNDING_BOX_COLOR,
    use_normalized_coordinates: bool = True,
):
    """
    Draw a bounding box to an image. Code loosely inspired by
    https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the 'use_normalized_coordinates' argument.
    Text is displayed on a separate line above the bounding box in black text on a rectangle filled with 'color'.
    If the top of the bounding box extends to the edge of the image, text is displayed below the bounding box.

    Args:
        image: a PIL.Image object.
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        text: strings to display in box.
        color: color to draw bounding box and text rectangle. Default is BOUNDING_BOX_COLOR.
        use_normalized_coordinates: If True (default), treat coordinates as relative to the image.
            Otherwise treat coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    line_thickness = 4 * int(np.ceil(0.001 * max(im_width, im_height)))
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    lines = [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)]
    draw.line(xy=lines, width=line_thickness, fill=color)
    if text != "" and text is not None:
        text_line_list = text.splitlines()
        scaled_font = scale_bounding_box_font(image, text_line_list, right - left)
        # If the total height of the display strings added to the top of the bounding box
        # exceeds the top of the image, stack the strings below the bounding box instead of above.
        text_height = sum([scaled_font.getsize(t)[1] for t in text_line_list])
        text_height_with_margin = (1 + 2 * 0.05) * text_height  # Each line has a top and bottom margin of 0.05x
        text_bottom = top
        if top < text_height_with_margin:
            text_bottom += text_height_with_margin
        # Reverse list and print from bottom to top.
        for t in text_line_list[::-1]:
            text_width, text_height = scaled_font.getsize(t)
            margin = int(np.ceil(0.05 * text_height))
            draw.rectangle(
                xy=[(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color
            )
            draw.text(xy=(left + margin, text_bottom - text_height - margin), text=t, fill="black", font=scaled_font)
            text_bottom -= text_height - 2 * margin
