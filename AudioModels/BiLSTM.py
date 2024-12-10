import pandas as pd
import numpy as np
import os
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# Ensure determinism in results
torch.manual_seed(0)
np.random.seed(0)

directory = "C:/Users/KlaraDaly/Desktop/csc4006-preliminary-code-main/Audio-Reading"
fold_csv_path = "C:/Users/KlaraDaly/Desktop/csc4006-preliminary-code-main/FoldLists/ReadingFolds.csv"
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

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=3):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.5, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Times 2 because of bidirectionality
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # Times 2 for bidirectionality
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        out = self.sigmoid(out)
        return out

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

# Function to calculate accuracy
def calculate_accuracy(voted_predictions, true_labels):
    correct_predictions = sum(1 for identifier, prediction in voted_predictions.items()
                              if prediction == true_labels[identifier])
    accuracy = correct_predictions / len(true_labels)
    return accuracy

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
    model = LSTMModel(X_train.shape[2], 64, num_layers=2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    
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