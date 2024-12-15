import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os


predictions = np.load('predictions.npy')  # Predicted labels
true_labels = np.load('true_labels.npy')  # True labels


accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average='weighted') 
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')


cm = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(cm)


def plot_roc_curve(true_labels, predictions, n_classes):
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        #  one-hot encoded true labels and predicted probabilities
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {i}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

# One-hot encode the predictions and true labels for ROC AUC calculation
n_classes = len(np.unique(true_labels))
true_labels_bin = label_binarize(true_labels, classes=np.arange(n_classes))
predictions_bin = label_binarize(predictions, classes=np.arange(n_classes))

plot_roc_curve(true_labels_bin, predictions_bin, n_classes)


# Validate test_data before prediction
test_data = np.array([])  # Replace with actual test data

if test_data.size == 0:
    raise ValueError("Test data is empty. Please ensure you have loaded the test data properly.")
elif len(test_data.shape) != 4:  
    raise ValueError(f"Test data has incorrect shape: {test_data.shape}. Expected shape is (samples, height, width, channels).")


test_dir = "path_to_test_images"  # Replace with test data directory
test_data = []

for filename in os.listdir(test_dir):
    img_path = os.path.join(test_dir, filename)
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img)
    test_data.append(img_array)

test_data = np.array(test_data)
print(f"Test data shape: {test_data.shape}")


if test_data.ndim == 3:  # If data is not in batch format, add batch dimension
    test_data = np.expand_dims(test_data, axis=0)

print(f"Test data after adjustment: {test_data.shape}")


np.save('predictions.npy', predictions)
np.save('true_labels.npy', true_labels)
