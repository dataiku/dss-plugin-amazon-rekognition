from PIL import Image, ImageDraw, ExifTags, ImageFont
import boto3

def aws_client(service_name, connection_info):
    return boto3.client(service_name=service_name, aws_access_key_id=connection_info.get('accessKey'), aws_secret_access_key=connection_info.get('secretKey'), region_name=connection_info.get('region'))


def draw_bounding_boxes(image_file, rekognition_response):
    objects_list = []
    for label in rekognition_response.get('Labels'):
        if label.get('Instances'):
            objects_list.append(label)

    image = Image.open(image_file)
    output_image = image.rotate(0, expand=True)
    width, height = output_image.size
    draw = ImageDraw.Draw(output_image)

    for object_type in objects_list:
        name = object_type.get('Name')
        for object_instance in object_type.get('Instances'):
            box = object_instance.get('BoundingBox')
            width, height = output_image.size
            left = width * box['Left']
            top = height * box['Top']
            draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])])
            draw.text([left, top], name) # add proba?
    return output_image


def generate_unique(name, existing_names):
    new_name = name
    for j in range(1, 1000):
        if new_name not in existing_names:
            return new_name
        new_name = name + "_{}".format(j)
    raise Exception("Failed to generated a unique name")
