import json
import logging
from PIL import Image
import dataiku
from dataiku.customrecipe import *
from dku_amazon_rekognition import *
from bbox import draw_bounding_boxes

#==============================================================================
# SETUP
#==============================================================================

logging.basicConfig(level=logging.INFO, format='[rekognition plugin] %(levelname)s - %(message)s')

connection_info = get_recipe_config().get('connectionInfo', {})
should_output_raw_results = get_recipe_config().get('should_output_raw_results')

input_folder_name = get_input_names_for_role('input_folder')[0]
input_folder = dataiku.Folder(input_folder_name)
input_folder_path = input_folder.get_path()

output_dataset_name = get_output_names_for_role('output_dataset')[0]
output_dataset = dataiku.Dataset(output_dataset_name)

client = get_client(connection_info)

#==============================================================================
# RUN
#==============================================================================

output_schema = [
    {"name": "file_path", "type": "string"},
    {"name": "is_adult_content", "type": "boolean"},
    {"name": "adult_score", "type": "double"},
    {"name": "is_suggestive_content", "type": "boolean"},
    {"name": "suggestive_score", "type": "double"},
    {"name": "is_violent_content", "type": "boolean"},
    {"name": "violence_score", "type": "double"}
]
if should_output_raw_results:
    output_schema.append({"name": "raw_results", "type": "string"})
output_dataset.write_schema(output_schema)

writer = output_dataset.get_writer()
for filepath in os.listdir(input_folder_path):
    if supported_image_format(filepath):
        with open(os.path.join(input_folder_path, filepath), "rb") as image_file:
            row, response = detect_adult_content(image_file, client)
            if should_output_raw_results:
                row["raw_results"] = json.dumps(response, default=lambda x: x.__dict__)
    else:
        logging.warn("Cannot score file (only JPEG, JPG and PNG extension are supported): " + filepath)
        row = {}
    row["file_path"] = filepath
    writer.write_row_dict(row)
writer.close()
