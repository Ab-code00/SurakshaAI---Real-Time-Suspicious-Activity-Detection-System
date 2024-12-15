import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def create_dir(directory):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def normalize_image(image):
    """Normalize the image by dividing by 255."""
    return image / 255.0

def augment_frame(image, angle=15, flip=True, brightness_variation=True):
    """Apply augmentation techniques such as rotation, flipping, and brightness adjustment."""
    # Rotation
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    # Flip image
    if flip:
        flipped_image = cv2.flip(rotated_image, 1)  
    else:
        flipped_image = rotated_image

    # Brightness adjustment
    if brightness_variation:
        hsv = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2HSV)
        value = np.random.uniform(0.5, 1.5)
        hsv[..., 2] = np.clip(hsv[..., 2] * value, 0, 255)  
        augmented_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        augmented_image = flipped_image

    return augmented_image

def preprocess_images(image_dir, output_dir, label, frame_size=(64, 64), label_list=None):
    """Preprocess images by applying resizing, normalization, and augmentation."""
    create_dir(output_dir)  # Create the main output directory if it doesn't exist

    # Check if the provided image directory exists
    if not os.path.exists(image_dir):
        print(f"Error: The directory {image_dir} does not exist.")
        return

    
    for image_file in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_file)
        
       
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image = cv2.imread(image_path)

        
        if image is None:
            print(f"Could not read image {image_file}")
            continue

        
        image = cv2.resize(image, frame_size)

      
        augmented_image = augment_frame(image)

 
        normalized_image = normalize_image(augmented_image)

        # Save the processed image
        processed_frame_path = os.path.join(output_dir, image_file)
        cv2.imwrite(processed_frame_path, normalized_image * 255)  # Convert back to 0-255 scale

        # Append the label and path to the list
        if label_list is not None:
            label_list.append((processed_frame_path, label))

    print(f"Preprocessed images saved to {output_dir}")

if __name__ == "__main__":
   
    categories =  ["Abuse","Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing",  "Vandalism", "NormalVideos"]
    base_image_directory = r"Raw"
    base_output_directory = r"data/processed"
    
   
    image_label_list = []

    for category in categories:
        image_directory = os.path.join(base_image_directory, category)
        output_directory = os.path.join(base_output_directory, category)

      
        label = 0 if category == "NormalVideos" else 1

        preprocess_images(image_directory, output_directory, label, label_list=image_label_list)

    # Save labels to a CSV file
    df = pd.DataFrame(image_label_list, columns=["image_path", "label"])
    df.to_csv(os.path.join(base_output_directory, "image_labels.csv"), index=False)
    print(f"Labels saved to {os.path.join(base_output_directory, 'image_labels.csv')}")