from yolo_onnx.yolov8_onnx import YOLOv8
import json
import base64
from io import BytesIO
from PIL import Image
import logging

# Initialize YOLOv8 object detector
yolov8_detector = YOLOv8('./models/yolov8s.onnx')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main(event, context):

    logger.debug("Event received: %s", event)

    # get payload
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
