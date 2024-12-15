# real_time_detection.py

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

# Real-time detection function
def real_time_detection(source=0, conf_threshold=0.5, nms_threshold=0.4):
    # Open video capture (0 for webcam or provide video file path)
    cap = cv2.VideoCapture(source)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        # Create blob and forward pass
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True, crop=False)
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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow("Real-Time Object Detection", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
real_time_detection()  # Use 0 for webcam; replace with file path for video input