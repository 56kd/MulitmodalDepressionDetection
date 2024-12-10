from collections import Counter
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

# Ensure determinism in results
torch.manual_seed(0)
np.random.seed(0)

directory = "C:/Users/44746/Desktop/Project/Audio-Interview"
fold_csv_path = "C:/Users/44746/Desktop/Project/InterviewFolds.csv"
fold_info = pd.read_csv(fold_csv_path)
fold_files = [fold_info[fold].dropna().tolist() for fold in fold_info.columns]
output_file_path = "C:/Users/44746/Desktop/Project/audio_it_pred_cnnbi.txt"

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

class CNNRNNModel(nn.Module):
    def __init__(self, num_channels, hidden_dim, num_layers, bidirectional):
        super(CNNRNNModel, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional 
        cnn_output_size = 64 * 64
        self.lstm = nn.LSTM(cnn_output_size, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)  # Adjust for bidirectional
        self.sigmoid = nn.Sigmoid()
        
        # Attention layer
        self.attention_layer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = x.view(x.size(0), 1, -1)  # Ready for LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.bidirectional:
            # Process bidirectional outputs
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # Apply attention
        attention_weights = F.softmax(self.attention_layer(lstm_out).squeeze(2), dim=1)
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        
        out = self.fc(context_vector)
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
        vote_result = np.round(np.mean(preds))
        final_predictions[identifier] = int(vote_result)
    return final_predictions

def extract_true_labels(identifiers):
    return {identifier: 1 if 'PM' in identifier or 'PF' in identifier else 0 for identifier in identifiers}

all_y_trues, all_y_preds = [], []
temp, temp1 = [], []

for fold_idx in range(len(fold_files)):
    fold_accuracy = 0
    fold_f1 = 0
    fold_precision = 0
    fold_recall = 0
    conf_matrix = confusion_matrix(temp, temp1)

    test_filenames = fold_files[fold_idx]
    train_filenames = [item for sublist in fold_files[:fold_idx] + fold_files[fold_idx+1:] for item in sublist]
    
    X_test, y_test, test_identifiers = load_features_for_fold(directory, test_filenames)
    X_train, y_train, _ = load_features_for_fold(directory, train_filenames)

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

    model = CNNRNNModel(num_channels=32, hidden_dim=128, num_layers=2, bidirectional=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Training loop
    model.train()
    for epoch in range(15): # 10 for RT, 15 for IT yields best performance
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor).squeeze().numpy()  # Get raw probabilities

    # Convert to list if needed, but keep as numpy array for efficiency
    predictions = outputs

    # Aggregate predictions by file, calculating average probability per file
    file_predictions = {}
    for identifier, prediction in zip(test_identifiers, predictions):
        if identifier not in file_predictions:
            file_predictions[identifier] = []
        file_predictions[identifier].append(prediction)

    file_avg_probability = {filename: np.mean(probs) for filename, probs in file_predictions.items()}

    # Open the file in append mode
    with open(output_file_path, 'a') as file:
        for filename, avg_prob in file_avg_probability.items():
            # Predicted label based on the average probability
            predicted_label = '1' if avg_prob > 0.5 else '0'
            print(f"Filename: {filename}, Average Probability: {avg_prob:.4f}, Predicted Label: {predicted_label}")
            # Write the filename and predicted label to file
            file.write(f"{filename}, {predicted_label}, {avg_prob}\n")

    # If you still need to perform operations with true labels and "voted" predictions:
    true_labels_dict = extract_true_labels(test_identifiers)
    all_y_trues = [true_labels_dict[identifier] for identifier in set(test_identifiers)]
    all_y_preds = [round(file_avg_probability[identifier]) for identifier in set(test_identifiers)]

    # Calculate and print fold metrics using all_y_trues and all_y_preds
    fold_accuracy = accuracy_score(all_y_trues, all_y_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_y_trues, all_y_preds, average='binary')
    precision *= 100
    recall *= 100
    f1 *= 100

    print(f"Fold Metrics: Accuracy: {fold_accuracy:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1 Score: {f1:.2f}%")

    # Compute the confusion matrix for the fold
    conf_matrix = confusion_matrix(all_y_trues, all_y_preds)
    print("Confusion Matrix:")
    print(conf_matrix)
