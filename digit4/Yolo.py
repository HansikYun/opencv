import cv2
import numpy as np

net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames() 

with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

image = cv2.imread('./image/sheep.png')
height, width = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

outs = net.forward(output_layers)

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5 and classes[class_id] == "sheep":
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
