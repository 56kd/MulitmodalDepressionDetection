import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load the txt file
df = pd.read_csv('audio_rt_predictions.txt', header=None, names=['filename', 'predicted_label', 'probability'])

# Define a function to extract the true label from the filename
def extract_label(filename):
    if 'PM' in filename or 'PF' in filename:
        return 1
    elif 'CM' in filename or 'CF' in filename:
        return 0
    else:
        raise ValueError("Filename does not contain a valid label indicator.")

# Apply the function to create a new column for true labels
df['true_label'] = df['filename'].apply(extract_label)

# Calculate metrics
accuracy = accuracy_score(df['true_label'], df['predicted_label'])
precision = precision_score(df['true_label'], df['predicted_label'])
recall = recall_score(df['true_label'], df['predicted_label'])
f1 = f1_score(df['true_label'], df['predicted_label'])

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
