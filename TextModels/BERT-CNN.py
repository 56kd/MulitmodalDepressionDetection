from collections import Counter
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# Ensure determinism in results
torch.manual_seed(0)
np.random.seed(0)

directory = "C:/Users/44746/Desktop/Project/BERT-Interview"
fold_csv_path = "C:/Users/44746/Desktop/Project/InterviewFolds.csv"
fold_info = pd.read_csv(fold_csv_path)
fold_files = [fold_info[fold].dropna().tolist() for fold in fold_info.columns]

def load_bert_embeddings(directory, filenames):
    embeddings = []
    labels = []
    identifiers = []
    for identifier in filenames:
        filename = f"{identifier}_it_bert_features.csv"
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            feature_vector = pd.read_csv(path, header=None).values.flatten()
            embeddings.append(feature_vector)
            label = 1 if 'PM' in identifier or 'PF' in identifier else 0
            labels.append(label)
            identifiers.append(identifier)

        else:
            print(f"File not found: {path}")
    
    return np.array(embeddings), np.array(labels), identifiers

class CNN1DModel(nn.Module):
    def __init__(self, input_dim):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)

        self.flatten = nn.Flatten()
        reduced_dim = input_dim // 4  # Two pooling layers
        self.fc1 = nn.Linear(32 * reduced_dim, 1)

    def forward(self, x):
        x = self.pool1(self.bn1(torch.relu(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(self.bn2(torch.relu(self.conv2(x))))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

    
# Load embeddings and labels
all_y_trues, all_y_preds = [], []

for fold_idx in range(len(fold_files)):
    test_filenames = fold_files[fold_idx]
    train_filenames = [item for sublist in fold_files[:fold_idx] + fold_files[fold_idx+1:] for item in sublist]
    
    # Load embeddings and labels
    X_test, y_test, _ = load_bert_embeddings(directory, test_filenames)
    X_train, y_train, _ = load_bert_embeddings(directory, train_filenames)

    # Convert to PyTorch tensors and reshape for Conv1D: (batch, channels, length)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Verify dimensions
    print("X_train_tensor shape:", X_train_tensor.shape)  # Should be [n_samples, 1, n_features]
    print("X_test_tensor shape:", X_test_tensor.shape)  # Should be [n_samples, 1, n_features]

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=32)

    input_dim = X_train_tensor.shape[2]  # Correctly accessing the number of features
    model = CNN1DModel(input_dim)

    # Set up weights and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Adjust weights to balance recall between classes
    weights = torch.tensor([2.0], dtype=torch.float32)  # Example adjustment for binary classification
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)


    # Training loop
    model.train()
    for epoch in range(100):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (outputs.squeeze() > 0.5).float().numpy()

    all_y_trues.extend(y_test)
    all_y_preds.extend(predictions)

    print(f"Fold {fold_idx + 1} Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Not Depressed', 'Depressed'], digits=4))

print("Overall Classification Report:")
print(classification_report(all_y_trues, all_y_preds, target_names=['Not Depressed', 'Depressed'], digits=4))

overall_accuracy = np.mean(np.array(all_y_trues) == np.array(all_y_preds))
print(f"Overall Accuracy: {overall_accuracy:.4f}")