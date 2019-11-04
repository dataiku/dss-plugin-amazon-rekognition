# Amazon Recognition Plugin


## Plugin information

This Dataiku DSS plugin provides several tools to interact with [Amazon Rekognition](https://aws.amazon.com/rekognition/), the Computer Vision API.

For example, it can be used to detect objects in an image, or to find all the faces in the image.

## Using the Plugin

### Prerequisites
In order to use the Plugin, you will need:

* an AWS account
* proper [credentials](https://docs.aws.amazon.com/comprehend/latest/dg/setting-up.html) (access tokens) to interact with the service:
* make sure you know in **which AWS region the services are valid**, the Plugin will need this information to get authenticated

### Plugin components
The Plugin has the following components:

    * [Object Detection](https://docs.aws.amazon.com/rekognition/latest/dg/labels.html):
    this operation detects objects in the image.
