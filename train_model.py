# Import necessary libraries
import os
import cv2
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
PROCESSED_DIR = r"data/processed"
BATCH_SIZE = 8 
EPOCHS = 1  
IMG_SIZE = (224, 224)
TIMESTEPS = 10  
INPUT_SHAPE = (TIMESTEPS, *IMG_SIZE, 3)  
CATEGORIES = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", 
              "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", 
              "Stealing", "Vandalism", "NormalVideos"]
TEST_SIZE = 0.2 

# Ensure the checkpoint directory exists
checkpoint_dir = 'models'
os.makedirs(checkpoint_dir, exist_ok=True)


def split_data(processed_dir, categories, test_size=0.2):
    data_paths = []
    labels = []
    for folder_name in categories:
        folder_path = os.path.join(processed_dir, folder_name)
        if os.path.isdir(folder_path):
            image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".png")]
            data_paths.extend(image_files)
            labels.extend([folder_name] * len(image_files))
    
 
    train_paths, test_paths, train_labels, test_labels = train_test_split(data_paths, labels, test_size=test_size, stratify=labels)
    return (train_paths, train_labels), (test_paths, test_labels)


def data_generator(data_paths, data_labels, categories, img_size=(224, 224), batch_size=4, timesteps=10):
    label_encoder = LabelEncoder()
    label_encoder.fit(categories) 

    while True:
        X_batch, y_batch = [], []
        batch_indices = np.random.choice(len(data_paths), size=batch_size, replace=False)

        for idx in batch_indices:
            sequence = []
            image_path = data_paths[idx]
            label = data_labels[idx]

            # Load frames in sequence for the video clip
            image = cv2.imread(image_path)
            if image is None:
                continue  
            image = cv2.resize(image, img_size)
            sequence.append(image)  
            
            while len(sequence) < timesteps:
                sequence.append(image)  

            X_batch.append(sequence)
            y_batch.append(label)

            if len(X_batch) == batch_size:
                X_batch = np.array(X_batch).astype('float32') / 255.0  # Normalize the data
                y_batch_encoded = label_encoder.transform(y_batch)  
                y_batch_categorical = to_categorical(y_batch_encoded, num_classes=len(categories))  # One-hot encoding
                yield X_batch, y_batch_categorical 
                X_batch, y_batch = [], []  

# Build the model
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    
    model.add(TimeDistributed(Flatten()))
    
    model.add(LSTM(64, return_sequences=False)) 
    model.add(Dropout(0.5)) 
    model.add(Dense(len(CATEGORIES), activation='softmax')) 
    
    return model


with tf.device('/GPU:0'):
    model = build_model(INPUT_SHAPE)
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Save the best model during training
checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, 'hybrid.keras'), monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  


(train_paths, train_labels), (test_paths, test_labels) = split_data(PROCESSED_DIR, CATEGORIES, TEST_SIZE)

# Train the model
train_generator = data_generator(train_paths, train_labels, CATEGORIES, img_size=IMG_SIZE, batch_size=BATCH_SIZE, timesteps=TIMESTEPS)
test_generator = data_generator(test_paths, test_labels, CATEGORIES, img_size=IMG_SIZE, batch_size=BATCH_SIZE, timesteps=TIMESTEPS)
steps_per_epoch = len(train_paths) // BATCH_SIZE
validation_steps = len(test_paths) // BATCH_SIZE

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint, early_stopping]
)

print("Model training complete!")

# Optionally, plot the training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
