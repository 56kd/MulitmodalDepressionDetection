import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from collections import Counter
import math

# Ensure determinism in results
torch.manual_seed(0)
np.random.seed(0)

directory = "C:/Users/44746/Desktop/Project/Audio-Reading"
fold_csv_path = "C:/Users/44746/Desktop/Project/ReadingFolds.csv"
fold_info = pd.read_csv(fold_csv_path)
fold_files = [fold_info[fold].dropna().tolist() for fold in fold_info.columns]

def load_features_for_fold(directory, filenames):
    segments = []
    labels = []
    identifiers = []
    for identifier in filenames:
        filename = f"{identifier}_rt_audio_features.csv"
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            features = pd.read_csv(path).values
            label = 1 if 'PM' in identifier or 'PF' in identifier else 0
            L, step = 128, 64
            for start in range(0, len(features) - L + 1, step):
                segment = features[start:start+L]
                segments.append(segment)
                labels.append(label)
                identifiers.append(identifier)
    return np.array(segments), np.array(labels), identifiers

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead, num_layers, output_dim=1, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        output = self.fc(output[:, -1, :])
        output = self.sigmoid(output)
        return output

def majority_voting(predictions, identifiers):
    votes = {}
    for identifier, prediction in zip(identifiers, predictions):
        if identifier not in votes:
            votes[identifier] = []
        votes[identifier].append(prediction)
    final_predictions = {}
    for identifier, preds in votes.items():
        vote_result = Counter(preds).most_common(1)[0][0]
        final_predictions[identifier] = vote_result
    return final_predictions

def extract_true_labels(identifiers):
    true_labels = {}
    for identifier in identifiers:
        true_labels[identifier] = 1 if 'PM' in identifier or 'PF' in identifier else 0
    return true_labels

# Initialize lists for aggregated results
all_y_trues, all_y_preds = [], []

# Cross-validation loop
accuracies = []
for fold_idx in range(len(fold_files)):
    print(f"Processing fold {fold_idx + 1} as test set")
    test_filenames = fold_files[fold_idx]
    train_filenames = [item for sublist in fold_files[:fold_idx] + fold_files[fold_idx+1:] for item in sublist]
    
    # Load features and identifiers for training and testing
    X_test, y_test, test_identifiers = load_features_for_fold(directory, test_filenames)
    X_train, y_train, _ = load_features_for_fold(directory, train_filenames)
    
    # Convert data to PyTorch tensors and create DataLoader
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=32)
    
    # Instantiate the model for the current fold
    model = TransformerModel(input_dim=X_train.shape[2], hidden_dim=2048, nhead=8, num_layers=2, dropout=0.1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(100):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (outputs.squeeze() > 0.5).float().numpy()

    # Apply majority voting to aggregate predictions by recording
    voted_predictions = majority_voting(predictions, test_identifiers)
    
    # Extract true labels from identifiers and convert predictions for comparison
    true_labels = extract_true_labels(test_identifiers)
    y_true = [true_labels[id] for id in test_identifiers]
    y_pred = [voted_predictions[id] for id in test_identifiers]

    # Append aggregated results for overall metrics calculation
    all_y_trues.extend(y_true)
    all_y_preds.extend(y_pred)
    
    # Print fold classification report
    print(f"Fold {fold_idx + 1} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Not Depressed', 'Depressed']))

# Calculate and print overall classification report
print("Overall Classification Report:")
print(classification_report(all_y_trues, all_y_preds, target_names=['Not Depressed', 'Depressed']))

# Calculate overall accuracy
overall_accuracy = np.mean(np.array(all_y_trues) == np.array(all_y_preds))
print(f"Overall Accuracy: {overall_accuracy:.4f}")
