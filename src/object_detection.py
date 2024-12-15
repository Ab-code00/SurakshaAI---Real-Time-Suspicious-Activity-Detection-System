# object_detection.py

import cv2
import numpy as np

# Load YOLOv7 model
weights_path = 'yolov7.weights'
config_path = 'yolov7.cfg'
class_names_path = 'coco.names'

# Load class names
with open(class_names_path, 'r') as f:
    class_names = f.read().strip().split('\n')

# Load the YOLOv7 network
net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Detection function
def detect_objects(image_path, conf_threshold=0.5, nms_threshold=0.4):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Create blob and forward pass
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(net.getUnconnectedOutLayersNames())

    # Process detections
    boxes, confidences, class_ids = [], [], []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maxima suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = str(class_names[class_ids[i]])
        confidence = confidences[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save and display the image with detections
    cv2.imwrite('detected_image.jpg', image)
    cv2.imshow("Detected Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_objects('input_image.jpg')  # Replace 'input_image.jpg' with the path to your image