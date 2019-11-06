import json
import boto3

SUPPORTED_IMAGE_FORMATS = ['jpeg', 'jpg', 'png']

def supported_image_format(filepath):
    extension = filepath.split(".")[-1].lower()
    return extension in SUPPORTED_IMAGE_FORMATS

def get_client(connection_info):
    return boto3.client(service_name="rekognition", aws_access_key_id=connection_info.get('accessKey'), aws_secret_access_key=connection_info.get('secretKey'), region_name=connection_info.get('region'))


def detect_labels(image_file, client):
    row = {}
    response = client.detect_labels(Image={'Bytes': image_file.read()})
    labels = [l['Name'] for l in response.get("Labels")]
    if len(labels):
        row["detected_labels"] = json.dumps(labels)
    return row, response, labels

def detect_objects(image_file, client):
    row = {}
    response = client.detect_labels(Image={'Bytes': image_file.read()})
    bbox_list = _format_bounded_boxes(response.get("Labels"))
    if len(bbox_list):
        row["detected_objects"] = json.dumps(bbox_list)
    return row, response, bbox_list

def detect_adult_content(image_file, client):
    row = {"adult_score": 0, "suggestive_score": 0, "violence_score": 0}
    response = client.detect_moderation_labels(Image={'Bytes': image_file.read()})

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
            ret.append({
                "label": label["Name"],
                "score": instance["Confidence"],
                "top": instance["BoundingBox"]["Top"],
                "left": instance["BoundingBox"]["Left"],
                "width": instance["BoundingBox"]["Width"],
                "height": instance["BoundingBox"]["Height"]
            })
    return ret
