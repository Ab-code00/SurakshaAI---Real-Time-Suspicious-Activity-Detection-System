import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import time
import torch
from ultralytics import YOLO

# Set device (CUDA or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
PROCESSED_DIR = r"data/processed"
CATEGORIES = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", 
              "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", 
              "Stealing", "Vandalism", "NormalVideos"]
TIMESTEPS = 10  # Number of frames in a video sequence
IMG_SIZE = (224, 224)  # Size for input to original model
BATCH_SIZE = 32  # Number of sequences to process in each batch

# Load YOLOv8 model
yolo_model = YOLO('yolov8s.pt')  # Use YOLOv8 small (yolov8s) model

# Load your custom Keras model
model = tf.keras.models.load_model('models/hybrid.keras')

# Prepare test data
def prepare_test_data(processed_dir, categories):
    data_paths = []
    labels = []
    for folder_name in categories:
        folder_path = os.path.join(processed_dir, folder_name)
        if os.path.isdir(folder_path):
            image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".png")]
            data_paths.extend(image_files)
            labels.extend([folder_name] * len(image_files))
    
    return data_paths, labels

# Test data generator
def test_data_generator(data_paths, data_labels, img_size=(224, 224), timesteps=10, batch_size=32):
    while True:
        for start in range(0, len(data_paths), batch_size):
            end = min(start + batch_size, len(data_paths))
            batch_sequences = []
            batch_labels = []

            for idx in range(start, end):
                sequence = []
                image_path = data_paths[idx]
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Image at {image_path} could not be loaded.")
                    continue
                image_resized = cv2.resize(image, img_size)
                sequence.extend([image_resized] * timesteps)  # Repeating same frame for simplicity
                
                batch_sequences.append(np.array(sequence).astype('float32') / 255.0)  # Normalize the data
                batch_labels.append(data_labels[idx])

            yield np.array(batch_sequences), np.array(batch_labels)

# Load data paths and labels
test_paths, test_labels = prepare_test_data(PROCESSED_DIR, CATEGORIES)

# Label Encoder
label_encoder = LabelEncoder()
label_encoder.fit(CATEGORIES)

# Prepare test data generator
test_generator = test_data_generator(test_paths, test_labels, img_size=IMG_SIZE, timesteps=TIMESTEPS, batch_size=BATCH_SIZE)

# Initialize lists to store predictions
predictions = []
true_labels = []

# Timing the prediction process
start_time = time.time()

# Get number of batches
num_batches = len(test_paths) // BATCH_SIZE + 1
batch_time = []

# Iterate through the test generator and make predictions
for batch_num in range(num_batches):
    try:
        batch_start_time = time.time()
        X_test, y_batch = next(test_generator)

        # Get YOLOv8 predictions
        yolo_preds = []
        for image in X_test:
            # Convert to the format YOLO expects (BGR and resized)
            img_bgr = cv2.cvtColor(image.astype('float32'), cv2.COLOR_RGB2BGR)
            results = yolo_model(img_bgr)  # Perform inference with YOLOv8
            
            # Extract bounding boxes, labels, and confidences
            boxes = results[0].boxes.xywh.cpu().numpy()  # XYWH format
            confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
            labels = results[0].boxes.cls.cpu().numpy()  # Class labels

            # Store the predictions
            yolo_preds.append((boxes, confidences, labels))

        # Get predictions from your original model (TensorFlow Keras)
        preds = model.predict(X_test)

        # Store TensorFlow model predictions
        predictions.extend(np.argmax(preds, axis=1))
        true_labels.extend(label_encoder.transform(y_batch))  # Encode the batch labels

        # Log batch time
        batch_elapsed_time = time.time() - batch_start_time
        batch_time.append(batch_elapsed_time)

        if batch_num > 0: 
            average_time_per_batch = np.mean(batch_time)
            estimated_remaining_batches = num_batches - (batch_num + 1)
            estimated_remaining_time = average_time_per_batch * estimated_remaining_batches
            print(f"Estimated time remaining: {estimated_remaining_time:.2f} seconds")
    except StopIteration:
        break

# Convert predictions and true labels to numpy arrays
predictions = np.array(predictions)
true_labels = np.array(true_labels)

# Save predictions and true labels
np.save('predictions.npy', predictions)
np.save('true_labels.npy', true_labels)

# Print the total elapsed time for predictions
elapsed_time = time.time() - start_time
print(f"Total evaluation time: {elapsed_time:.2f} seconds")

# Print classification report
print("Classification Report:")
print(classification_report(true_labels, predictions, target_names=CATEGORIES))

# Print confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(true_labels, predictions)
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(10, 7))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(CATEGORIES))
plt.xticks(tick_marks, CATEGORIES, rotation=45)
plt.yticks(tick_marks, CATEGORIES)

# Normalize the confusion matrix and add labels
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
thresh = cm_normalized.max() / 2.0
for i, j in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
    plt.text(j, i, format(cm_normalized[i, j], '.2f'),
             horizontalalignment="center",
             color="white" if cm_normalized[i, j] > thresh else "black")

plt.ylabel('True label')

plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
