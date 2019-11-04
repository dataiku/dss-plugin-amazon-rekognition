import json
import logging
import pandas as pd
import dataiku
from dataiku.customrecipe import *
from dku_aws import draw_bounding_boxes, aws_client

ALLOWED_FORMATS = ['jpeg', 'jpg', 'png']

#==============================================================================
# SETUP
#==============================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='aws-machine-learning plugin %(levelname)s - %(message)s')

connection_info = get_recipe_config().get('connectionInfo', {})

input_folder_name = get_input_names_for_role('input-folder')[0]
input_folder = dataiku.Folder(input_folder_name)
input_folder_path = input_folder.get_path()

output_dataset_name = get_output_names_for_role('output-dataset')[0]
output_dataset = dataiku.Dataset(output_dataset_name)

output_folder_name = get_output_names_for_role('output-folder')[0]
output_folder = dataiku.Folder(output_folder_name)
output_folder_path = output_folder.get_path()

#==============================================================================
# RUN
#==============================================================================

rekognition = aws_client('rekognition', connection_info)

output_rows = []
for filepath in os.listdir(input_folder_path):
    extension = filepath.split(".")[-1].lower()
    if extension in ALLOWED_FORMATS:
        fullpath = os.path.join(input_folder_path, filepath)
        image_file = open(fullpath,'rb')
        image_in_bytes = image_file.read()
        response = rekognition.detect_labels(Image={'Bytes': image_in_bytes})

        # TODO if output dataset
        row = {"file": filepath, "detected_object": json.dumps(response)}
        output_rows.append(row)

        # TODO if output folder
        image_with_bounding_boxes = draw_bounding_boxes(image_file, response)

        output_fullpath = os.path.join(output_folder_path, '.'.join(filepath.split(".")[:-1])+'_scored.'+extension)
        image_with_bounding_boxes.save(output_fullpath)
    else:
        logger.info("Cannot score file (only JPEG, JPG and PNG extension are supported): " + filepath)

# TODO if output dataset
output_dataset.write_with_schema(pd.DataFrame(output_rows))
