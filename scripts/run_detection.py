# scripts/run_detection.py

import cv2
import numpy as np
import tensorflow as tf

def load_models(hybrid_model_path):
    """Load the models."""
    hybrid_model = tf.keras.models.load_model(hybrid_model_path)
    return hybrid_model

def preprocess_frame(frame):
    """Preprocess the video frame for the model."""
    frame_resized = cv2.resize(frame, (64, 64))  # Resize as per your model input
    frame_normalized = frame_resized / 255.0  # Normalize
    return frame_normalized

def run_detection(video_source, hybrid_model):
    """Run detection on video stream."""
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)
        processed_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension

        # Make predictions
        hybrid_prediction = hybrid_model.predict(processed_frame)
        hybrid_class = np.argmax(hybrid_prediction)

        # Display results
        cv2.putText(frame, f'Predicted Class: {hybrid_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    hybrid_model_path = '../models/hybrid_best.keras'
    video_source = 0  # Use 0 for webcam or provide a video file path

    hybrid_model = load_models(hybrid_model_path)
    run_detection(video_source, hybrid_model)