from yolo_onnx.yolov8_onnx import YOLOv8
import json
import base64
from io import BytesIO
from PIL import Image
import logging
import boto3
import urllib

# Initialize S3 client
s3 = boto3.client('s3')

# Initialize YOLOv8 object detector
yolov8_detector = YOLOv8('./models/yolov8s.onnx')

logger = logging.getLogger()
logger.setLevel("INFO")

def main(event, context):

    logger.info("Event received: %s", event)
    print("Event received: ", event)

    if 'body' in event:
        # get payload
        try: 
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']

            # get params
            img_b64 = body['image']
            size = body.get('size', 640)
            conf_thres = body.get('conf_thres', 0.3)
            iou_thres = body.get('iou_thres', 0.5)

            # open image
            img = Image.open(BytesIO(base64.b64decode(img_b64.encode('ascii'))))

            # infer result
            detections = yolov8_detector(img, size=size, conf_thres=conf_thres, iou_thres=iou_thres)

            # return result
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "detections": detections
                }),
            }
        except Exception as e:
            logger.error("Error: %s", e)
    else:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
        
        try: 
            response = s3.get_object(Bucket=bucket, Key=key)
            logger.info("S3 response: %s", response)
        except Exception as e:
            logger.error("Error: %s", e)
            raise e
