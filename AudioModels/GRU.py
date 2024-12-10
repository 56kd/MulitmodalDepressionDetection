from collections import Counter
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report

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
    epsilon = 1e-8  # Small constant to prevent division by zero

    for identifier in filenames:
        filename = f"{identifier}_it_audio_features.csv" # it = Interview Task, rt = Interview Task
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            features = pd.read_csv(path).values

            label = 1 if 'PM' in identifier or 'PF' in identifier else 0
            L, step = 128, 64

            for start in range(0, len(features) - L + 1, step):
                segment = features[start:start+L]
                
                # Normalize the segment
                std_dev = np.std(segment, axis=0)
                zero_variance = std_dev == 0
                std_dev[zero_variance] = 1
                normalized_segment = (segment - np.mean(segment, axis=0)) / (std_dev + epsilon)
                
                segments.append(normalized_segment)
                labels.append(label)
                identifiers.append(identifier)

    return np.array(segments), np.array(labels), identifiers

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GRUModel, self).__init__()

        # GRU Layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # LSTM Layer
        # self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,)
        
        # Fully connected layer
        direction =  1
        self.fc = nn.Linear(hidden_dim * direction, 1)
        
        # Output layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gru_out, _ = self.gru(x)  # gru_out shape: [batch_size, sequence_length, hidden_dim]
        # Select the output of the last time step
        last_out = gru_out[:, -1, :]  # last_out shape: [batch_size, hidden_dim]
        out = self.fc(last_out)  # Shape: [batch_size, 1]
        out = self.sigmoid(out)
        return out
    
def majority_voting(predictions, identifiers):
    votes = {}
    for identifier, prediction in zip(identifiers, predictions):
        votes[identifier] = votes.get(identifier, []) + [prediction]
    final_predictions = {identifier: Counter(preds).most_common(1)[0][0] for identifier, preds in votes.items()}
    return final_predictions

def extract_true_labels(identifiers):
    return {identifier: 1 if 'PM' in identifier or 'PF' in identifier else 0 for identifier in identifiers}

def calculate_accuracy(voted_predictions, true_labels):
    correct_predictions = sum(1 for identifier, prediction in voted_predictions.items() if prediction == true_labels[identifier])
    return correct_predictions / len(true_labels)

all_y_trues, all_y_preds = [], []

for fold_idx in range(len(fold_files)):
    test_filenames = fold_files[fold_idx]
    train_filenames = [item for sublist in fold_files[:fold_idx] + fold_files[fold_idx+1:] for item in sublist]
    
    X_test, y_test, test_identifiers = load_features_for_fold(directory, test_filenames)
    X_train, y_train, _ = load_features_for_fold(directory, train_filenames)

    # Data augmentation (random noise injection)
    noise_factor = 0.005
    X_train += noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)

    X_train = X_train.transpose(0, 2, 1)
    X_test = X_test.transpose(0, 2, 1)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=32)

    num_channels = 128  # Number of input channels (e.g., number of features per time step)
    hidden_dim = 128   # Number of features in the hidden state of the LSTM
    num_layers = 1     # Number of recurrent layers in the LSTM

    model =GRUModel(num_channels, hidden_dim, num_layers)

    # Define loss function, optimizer, etc.
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)


    # Training loop
    model.train()
    for epoch in range(100): 
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (outputs.squeeze() > 0.5).float().numpy()

    voted_predictions = majority_voting(predictions, test_identifiers)
    true_labels = extract_true_labels(test_identifiers)
    y_true = [true_labels[id] for id in test_identifiers]
    y_pred = [voted_predictions[id] for id in test_identifiers]

    all_y_trues.extend(y_true)
    all_y_preds.extend(y_pred)

    # Print fold classification report with four significant figures
    print(f"Fold {fold_idx + 1} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Not Depressed', 'Depressed'], digits=4))

# Print overall classification report with four significant figures
print("Overall Classification Report:")
print(classification_report(all_y_trues, all_y_preds, target_names=['Not Depressed', 'Depressed'], digits=4))

# Calculate overall accuracy and print with four significant figures
overall_accuracy = np.mean(np.array(all_y_trues) == np.array(all_y_preds))
print(f"Overall Accuracy: {overall_accuracy:.4f}")