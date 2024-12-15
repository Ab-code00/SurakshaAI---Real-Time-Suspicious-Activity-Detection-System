# SurakshaAI---Real-Time-Suspicious-Activity-Detection-System
SurakshaAI is a real-time AI-powered system for detecting suspicious activities like harassment, fighting, and vandalism using live video feeds. Built with YOLOv5 and CNN-LSTM models, it ensures timely alerts to authorities. Trained on UCF-Crime and AVENUE datasets, it enhances safety in public spaces and private properties.
---

## Features
- **Real-Time Detection**: Monitors live camera feeds and detects suspicious activities instantly.
- **Multi-Class Detection**: Classifies activities such as harassment, fighting, vandalism, and abuse.
- **Alert System**: Sends real-time notifications to authorities upon detecting suspicious activities.
- **Dataset-Based Training**: Trained on UCF-Crime, AVENUE Video Dataset, and Violent-Flows for high accuracy.

---

## Technologies Used
- **YOLOv8**: For real-time object and people detection.
- **CNN-LSTM**: For recognizing and classifying human activities.
- **Deep Learning Frameworks**: TensorFlow, PyTorch.
- **Backend**: Flask/Django (replace with your backend framework).
- **Frontend**: Minimalistic web interface for real-time monitoring.
- **Cloud**: Deployed on [your cloud provider, if applicable].
- **Alert System**: Email/SMS notifications (integrated with Twilio/SMTP).

---

### **Download and Preprocess the Dataset**  
To use SurakshaAI, you'll need to download and preprocess the dataset for training and testing. Follow these steps:  

1. **Download the Dataset**  
   - Download the UCF-Crime dataset from the official link:  
     [UCF-Crime Dataset Download](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset).  
   - Once downloaded, move the dataset file to the `datasets/` directory:  
     ```bash
     mkdir datasets
     mv <path-to-downloaded-dataset> datasets/
     ```  

2. **Run the Preprocessing Script**  
   - To prepare the data for training and testing, run the `preprocess.py` script. This script processes the raw dataset and saves it in a format suitable for the model.  
     ```bash
     python preprocess.py
     ```  

3. **Store Processed Data**  
   - The processed data will be automatically saved in the `datasets/processed/` directory. Ensure the preprocessing completes successfully before proceeding.
  
## **Usage**  
Once you have completed the dataset download and preprocessing, you can proceed with training the model.
