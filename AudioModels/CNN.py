import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import os
from collections import Counter

# Ensure determinism in results
torch.manual_seed(0)
np.random.seed(0)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define directory and read fold information
directory = "C:/Users/KlaraDaly/Desktop/csc4006-preliminary-code-main/Audio-Interview"
fold_csv_path = "C:/Users/KlaraDaly/Desktop/csc4006-preliminary-code-main/FoldLists/InterviewFolds.csv"
fold_info = pd.read_csv(fold_csv_path)
fold_files = [fold_info[fold].dropna().tolist() for fold in fold_info.columns]


def load_features_for_fold(directory, filenames):
    segments = []
    labels = []
    identifiers = []
    epsilon = 1e-8  # Small constant to prevent division by zero

    for identifier in filenames:
        filename = f"{identifier}_it_audio_features.csv"
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            features = pd.read_csv(path).values

            label = 1 if 'PM' in identifier or 'PF' in identifier else 0
            L, step = 128, 64

            for start in range(0, len(features) - L + 1, step):
                segment = features[start:start + L]

                # Normalize the segment
                std_dev = np.std(segment, axis=0)
                zero_variance = std_dev == 0
                std_dev[zero_variance] = 1  # Adjust standard deviation for zero variance features
                normalized_segment = (segment - np.mean(segment, axis=0)) / (std_dev + epsilon)

                segments.append(normalized_segment)
                labels.append(label)
                identifiers.append(identifier)

    return np.array(segments), np.array(labels), identifiers


class SimpleCNNModel(nn.Module):
    def __init__(self, num_channels):
        super(SimpleCNNModel, self).__init__()
        self.layer1 = nn.Conv1d(num_channels, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop_out = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 64, 1)  # Adjusted for simplified model
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.layer1(x)))
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x


def majority_voting(predictions, identifiers):
    """Aggregate predictions at the file level."""
    votes = {}
    for identifier, prediction in zip(identifiers, predictions):
        if identifier not in votes:
            votes[identifier] = []
        votes[identifier].append(prediction)
    final_predictions = {identifier: Counter(preds).most_common(1)[0][0] for identifier, preds in votes.items()}
    return final_predictions


def extract_true_labels(identifiers):
    """Return the true label for each file/identifier."""
    return {identifier: 1 if 'PM' in identifier or 'PF' in identifier else 0 for identifier in set(identifiers)}


all_y_trues, all_y_preds = [], []

for fold_idx in range(len(fold_files)):
    test_filenames = fold_files[fold_idx]
    train_filenames = [item for sublist in fold_files[:fold_idx] + fold_files[fold_idx + 1:] for item in sublist]

    X_test, y_test, test_identifiers = load_features_for_fold(directory, test_filenames)
    X_train, y_train, _ = load_features_for_fold(directory, train_filenames)

    # Data augmentation (random noise injection)
    noise_factor = 0.005
    X_train += noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)

    X_train = X_train.transpose(0, 2, 1)
    X_test = X_test.transpose(0, 2, 1)

    # Convert to PyTorch tensors and move to device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=32)

    num_channels = 32  # Adjust based on your data's actual number of channels
    model = SimpleCNNModel(num_channels=num_channels).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (outputs.squeeze() > 0.5).float().cpu().numpy()

    voted_predictions = majority_voting(predictions, test_identifiers)
    true_labels = extract_true_labels(test_identifiers)
    y_true = [true_labels[id] for id in test_filenames]  # File-level ground truth
    y_pred = [voted_predictions[id] for id in test_filenames]  # File-level predictions

    all_y_trues.extend(y_true)
    all_y_preds.extend(y_pred)

    print(f"Fold {fold_idx + 1} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Not Depressed', 'Depressed'], digits=4))

# Calculate and print overall metrics
accuracy = accuracy_score(all_y_trues, all_y_preds)
precision = precision_score(all_y_trues, all_y_preds)
recall = recall_score(all_y_trues, all_y_preds)
f1 = f1_score(all_y_trues, all_y_preds)

print("\nOverall Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
