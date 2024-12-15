import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class ActivityRecognitionDataset(Dataset):
    def _init_(self, features_dir):
        self.features_dir = features_dir
        self.feature_files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]

    def _len_(self):
        return len(self.feature_files)

    def _getitem_(self, idx):
        feature_file = os.path.join(self.features_dir, self.feature_files[idx])
        features = np.load(feature_file)
        label = self.get_label_from_filename(self.feature_files[idx])
        return torch.tensor(features, dtype=torch.float32), label

    def get_label_from_filename(self, filename):
        # Assuming the label is encoded in the filename (you can modify this logic)
        return filename.split('_')[0]

class ActivityRecognizer(nn.Module):
    def _init_(self, input_size, num_classes):
        super(ActivityRecognizer, self)._init_()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last output
        return out

def train_activity_recognizer(features_dir, num_classes, epochs=10, batch_size=32):
    dataset = ActivityRecognitionDataset(features_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ActivityRecognizer(input_size=features_dir, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for features, labels in tqdm(dataloader):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')

if _name_ == "_main_":
    features_dir = r"data/features"  # Directory containing extracted features
    num_classes = 6  # Update this according to your dataset
    train_activity_recognizer(features_dir, num_classes)