from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

def load_predictions_and_probabilities(filename):
    """Load predictions and probabilities from a file, handling file not found errors."""
    predictions = {}
    probabilities = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    identifier, prediction, probability = parts
                    predictions[identifier] = int(float(prediction))
                    probabilities[identifier] = float(probability)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return {}, {}  # Return empty dictionaries on error
    return predictions, probabilities

def decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities, audio_weight=0.5, text_weight=0.5):
    """Fuses audio and text predictions based on weighted probabilities, handling missing identifiers and invalid weights."""
    if not (0 <= audio_weight <= 1) or not (0 <= text_weight <= 1):
        raise ValueError("Weights must be between 0 and 1.")

    fused_predictions = {}
    for identifier in audio_predictions:
        if identifier not in text_predictions:
            print(f"Warning: Missing identifier {identifier} in text predictions.")
            continue

        audio_pred = audio_predictions[identifier]
        text_pred = text_predictions[identifier]
        
        if audio_pred == text_pred:
            fused_predictions[identifier] = audio_pred
        else:
            audio_prob = audio_probabilities[identifier]
            text_prob = text_probabilities[identifier]
            weighted_avg_prob = ((audio_weight * audio_prob) + (text_weight * text_prob)) / (audio_weight + text_weight)
            fused_prediction = 1 if weighted_avg_prob > 0.5 else 0
            fused_predictions[identifier] = fused_prediction
    return fused_predictions

def save_fused_predictions(fused_predictions, filename='weightedaverage_it_test.txt'):
    """Save the fused predictions to a file, handling possible exceptions during file operations."""
    if not fused_predictions:
        raise ValueError("No predictions available to save.")

    try:
        with open(filename, 'w') as file:
            for identifier, prediction in fused_predictions.items():
                file.write(f"{identifier},{prediction}\n")
        print(f"Fused predictions saved to {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")
        raise

def evaluate_predictions(filename='weightedaverage_it_test.txt'):
    """Evaluate the predictions against true labels and print accuracy, precision, recall, and F1-score."""
    y_true = []
    y_pred = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) != 2:
                    print(f"Malformed line: {line.strip()}")
                    continue
                identifier, prediction = parts
                true_label = 1 if 'PM' in identifier or 'PF' in identifier else 0
                y_true.append(true_label)
                y_pred.append(int(prediction))

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    except FileNotFoundError:
        print(f"Error: File {filename} not found for evaluation.")

if __name__ == '__main__':
    # Define file paths and weights
    audio_file = 'C:/Users/44746/Desktop/Project/audio_it_predictions.txt'
    text_file = 'C:/Users/44746/Desktop/Project/text_it_predictions.txt'
    audio_weight = 0.8103
    text_weight = 0.9310

    # Load predictions
    audio_predictions, audio_probabilities = load_predictions_and_probabilities(audio_file)
    text_predictions, text_probabilities = load_predictions_and_probabilities(text_file)

    # Perform fusion
    fused_predictions = decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities, audio_weight, text_weight)

    # Save and evaluate predictions
    save_fused_predictions(fused_predictions)
    evaluate_predictions()