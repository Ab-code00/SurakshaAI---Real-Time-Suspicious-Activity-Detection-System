import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import torch

# Set device (CUDA or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLOv8 model
yolo_model = YOLO('yolov8s.pt')  # Load YOLOv8 Small model

# Load your custom Keras model
model = tf.keras.models.load_model('models/hybrid.keras')  # Your trained model

# Define the categories for classification

CATEGORIES = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", 
              "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", 
              "Stealing", "Vandalism", "NormalVideos"]

# Label Encoder for classification
label_encoder = LabelEncoder()
label_encoder.fit(CATEGORIES)

# Email alert setup
def send_alert(subject, message, recipient_email="abdulganishaikh4444@gmail.com"):
    """Send an alert email to the property owner."""
    sender_email =  "shaikhabdulgani997@gmail.com"  # Replace with your email

    sender_password = "wixp oemk atcx tdnj" # Replace with your email's app password

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print(f"Alert sent to {recipient_email}")
    except Exception as e:
        print(f"Failed to send alert: {str(e)}")

# Global list to store the frames for classification
frame_buffer = []  # Buffer to store the last 10 frames

# Function to process video frame for classification
def process_frame_for_classification(frame, model, img_size=(224, 224)):
    """Preprocess frame and predict using the TensorFlow classification model."""
    img_resized = cv2.resize(frame, img_size)
    img_normalized = img_resized.astype('float32') / 255.0
    frame_buffer.append(img_normalized)

    if len(frame_buffer) > 10:
        frame_buffer.pop(0)  # Remove the oldest frame if there are more than 10 frames

    if len(frame_buffer) == 10:  # Only classify when there are 10 frames in the buffer
        sequence = np.array(frame_buffer)
        sequence_expanded = np.expand_dims(sequence, axis=0)
        prediction = model.predict(sequence_expanded, verbose=0)
        predicted_class_idx = np.argmax(prediction, axis=1)
        predicted_class = label_encoder.inverse_transform(predicted_class_idx)
        return predicted_class[0]
    return None  # If not enough frames, return None

# Function to process frame for YOLO object detection
def process_frame_for_yolo(frame, yolo_model):
    """Perform YOLOv8 inference on the frame and get bounding boxes and class labels."""
    img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = yolo_model(img_bgr)
    
    boxes = results[0].boxes.xywh.cpu().numpy()  # Box coordinates
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    labels = results[0].boxes.cls.cpu().numpy()  # Class labels
    
    return boxes, confidences, labels

# Function to display results (bounding boxes and classification result)
def display_results(frame, boxes, confidences, labels, class_name):
    """Display bounding boxes, confidences, labels and classification result on frame."""
    for box, conf, label in zip(boxes, confidences, labels):
        if conf > 0.5:
            x1, y1, w, h = box
            label_name = yolo_model.names[int(label)]
            cv2.rectangle(frame, (int(x1 - w / 2), int(y1 - h / 2)), 
                          (int(x1 + w / 2), int(y1 + h / 2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} ({conf:.2f})", 
                        (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (255, 0, 0), 2)
    
    if class_name is not None:
        cv2.putText(frame, f"Event: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Live Crime Detection", frame)

# Start webcam feed
cap = cv2.VideoCapture(0)  # 0 for webcam, or provide a path to a video file

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Step 1: Perform object detection with YOLO
    boxes, confidences, labels = process_frame_for_yolo(frame, yolo_model)

    # Step 2: Classify the scene/event using the TensorFlow model
    class_name = process_frame_for_classification(frame, model)

    # Step 3: Display YOLO object detection results and classification result
    display_results(frame, boxes, confidences, labels, class_name)

    # Step 4: Send alert if a crime is detected
    if class_name and class_name not in ["NormalVideos"]:
        subject = f"Alert: {class_name} detected!"
        message = f"A potential crime ({class_name}) has been detected on the live feed."
        send_alert(subject, message)

    # Step 5: Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
