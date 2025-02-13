import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os

# Ensure reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Define a function to load BERT embeddings
def load_bert_embeddings(directory, filenames):
    embeddings = []
    labels = []
    identifiers = []
    for identifier in filenames:
        filename = f"{identifier}_rt_bert_features.csv"
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            feature_matrix = pd.read_csv(path, header=None).values
            embeddings.append(feature_matrix)
            label = 1 if 'PM' in identifier or 'PF' in identifier else 0
            labels.append(label)
            identifiers.append(identifier)
        else:
            print(f"File not found: {path}")
    return np.array(embeddings), np.array(labels), identifiers

# Paths to data
directory = "C:/Users/44746/Desktop/Project/BERT-Reading"
fold_csv_path = "C:/Users/44746/Desktop/Project/ReadingFolds.csv"

# Read the CSV file containing fold information
fold_info = pd.read_csv(fold_csv_path)
fold_files = [fold_info[fold].dropna().tolist() for fold in fold_info.columns]

# Initialize lists to store results
all_y_trues, all_y_preds = [], []

# Train and evaluate the model for each fold
for fold_idx in range(len(fold_files)):
    train_filenames = [item for sublist in fold_files[:fold_idx] + fold_files[fold_idx+1:] for item in sublist]
    test_filenames = fold_files[fold_idx]
    
    # Load the embeddings
    X_train, y_train, _ = load_bert_embeddings(directory, train_filenames)
    X_test, y_test, _ = load_bert_embeddings(directory, test_filenames)

    # Convert data to PyTorch tensors
    # Ensure the data is in the format: (batch_size, seq_len, features)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Determine the sequence length and input dimension
    seq_len = X_train_tensor.shape[1]  # Number of timesteps per sample
    input_dim = X_train_tensor.shape[2]  # Number of features per timestep
    
    # Reshape tensors to fit the LSTM input requirements
    X_train_tensor = X_train_tensor.view(-1, seq_len, input_dim)
    X_test_tensor = X_test_tensor.view(-1, seq_len, input_dim)
    
    
    # Create DataLoader instances
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Define the dimensions for the LSTM
    input_dim = X_train_tensor.shape[2]
    hidden_dim = 128
    model = LSTMModel(input_dim, hidden_dim)

    # Define loss function and optimizer
    weights = torch.tensor([2.0], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model.train()
    for epoch in range(100):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (outputs.squeeze() > 0.5).float().numpy()

    # Store results
    all_y_trues.extend(y_test)
    all_y_preds.extend(predictions)

    # Print classification report for the current fold
    print(f"Fold {fold_idx + 1} Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Not Depressed', 'Depressed'], digits=4))

# Print overall classification report
print("Overall Classification Report:")
print(classification_report(all_y_trues, all_y_preds, target_names=['Not Depressed', 'Depressed'], digits=4))

# Calculate and print overall accuracy
overall_accuracy = np.mean(np.array(all_y_trues) == np.array(all_y_preds))
print(f"Overall Accuracy: {overall_accuracy:.4f}")
