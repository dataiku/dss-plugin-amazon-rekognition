# -*- coding: utf-8 -*-
import os
import logging
from typing import AnyStr, List

import numpy as np
from dataiku.customrecipe import get_recipe_resource
from PIL import Image, ImageFont, ImageDraw


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

BOUNDING_BOX_COLOR = "red"
BOUNDING_BOX_TEXT_FONT_PATH = os.path.join(get_recipe_resource(), "SourceSansPro-Regular.ttf")


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def draw_bounding_box_pil_image(
    image: Image,
    ymin: float,
    xmin: float,
    ymax: float,
    xmax: float,
    display_str_list: List[AnyStr] = [],
    color: AnyStr = BOUNDING_BOX_COLOR,
    use_normalized_coordinates: bool = True,
):
    """
    Draw a bounding box to an image. Code adapted from
    https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py
    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.
    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
        image: a PIL.Image object.
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        display_str_list: list of strings to display in box (each to be shown on its own line).
        color: color to draw bounding box. Default is BOUNDING_BOX_COLOR.
        use_normalized_coordinates: If True (default), treat coordinates as relative to the image.
            Otherwise treat coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    thickness = int(0.004 * max(im_width, im_height))  # reasonable design heuristic
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype(font=BOUNDING_BOX_TEXT_FONT_PATH, size=24)
    except IOError:
        logging.warn("Bounding box text font could not be loaded")
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
        text_bottom -= text_height - 2 * margin
