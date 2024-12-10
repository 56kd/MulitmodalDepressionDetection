import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure determinism in results
torch.manual_seed(0)
np.random.seed(0)

directory = "C:/Users/44746/Desktop/Project/Audio-Interview"
fold_csv_path = "C:/Users/44746/Desktop/Project/InterviewFolds.csv"
fold_info = pd.read_csv(fold_csv_path)
fold_files = [fold_info[fold].dropna().tolist() for fold in fold_info.columns]

def load_features_for_fold(directory, filenames):
    segments = []
    labels = []
    identifiers = []
    for identifier in filenames:
        filename = f"{identifier}_it_audio_features.csv"
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
        self.num_layers = num_layers  # Ensure this is correctly initialized
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.attention_layer = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Apply attention
        attention_weights = F.softmax(self.attention_layer(lstm_out).squeeze(2), dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        
        out = self.fc(context_vector)
        out = self.sigmoid(out)
        return out

def majority_voting(predictions, identifiers):
    votes = {}
    for identifier, prediction in zip(identifiers, predictions):
        votes.setdefault(identifier, []).append(prediction)
    final_predictions = {identifier: round(np.mean(preds)) for identifier, preds in votes.items()}
    return final_predictions

def extract_true_labels(identifiers):
    return {identifier: 1 if 'PM' in identifier or 'PF' in identifier else 0 for identifier in identifiers}

# Cross-validation loop
overall_results = []

for fold_idx in range(len(fold_files)):
    print(f"Processing fold {fold_idx + 1} as test set")
    test_filenames = fold_files[fold_idx]
    train_filenames = [item for sublist in fold_files[:fold_idx] + fold_files[fold_idx+1:] for item in sublist]
    
    X_test, y_test, test_identifiers = load_features_for_fold(directory, test_filenames)
    X_train, y_train, _ = load_features_for_fold(directory, train_filenames)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=32)
    
    model = LSTMModel(X_train.shape[2], 64, num_layers=3)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
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
    
    # File-level majority voting
    voted_predictions = majority_voting(predictions, test_identifiers)
    true_labels = extract_true_labels(test_identifiers)
    
    # Aggregating results per file
    y_true = [true_labels[id] for id in set(test_identifiers)]
    y_pred = [voted_predictions[id] for id in set(test_identifiers)]
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=1)
    overall_results.append((acc, prec, rec, f1))
    print(f"Fold {fold_idx + 1} Metrics: Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

# Calculate overall metrics
overall_metrics = np.mean(overall_results, axis=0)
print(f"Overall Metrics: Accuracy: {overall_metrics[0]:.4f}, Precision: {overall_metrics[1]:.4f}, Recall: {overall_metrics[2]:.4f}, F1 Score: {overall_metrics[3]:.4f}")