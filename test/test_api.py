import requests
import cv2
import base64
import os

PATH = os.path.dirname(os.path.abspath(__file__))

IMG_PATH = os.path.join(PATH, '1605820957 (1).png')
with open(os.path.join(PATH, 'lambda_api_url.txt'), 'r') as file:
    API_URL = file.readline().strip()



# encode image to b64
with open(IMG_PATH, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode('ascii')

# trigger api
result = requests.get(API_URL, json={"image": img_b64})
print("API response received")
# extract detections

detections = result.json()['detections']
people_detection = [det for det in detections if det['class_id'] == 0]
people_count = len(people_detection)
print(f"Number of people detected: {people_count}")
# display detections
img = cv2.imread(IMG_PATH)
for det in people_detection:
    x0,y0,x1,y1 = det['bbox']
    img = cv2.rectangle(img, (x0,y0), (x1,y1), (0,0,255), 4)
cv2.imwrite(os.path.join(PATH, 'output.png'), img)
