from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

def load_predictions_and_probabilities(filename):
    # Load predictions and probabilities from a file.
    predictions = {}
    probabilities = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 3:
                identifier, prediction, probability = parts
                predictions[identifier] = int(float(prediction))
                probabilities[identifier] = float(probability)
    return predictions, probabilities

def decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities):
    fused_predictions = {}
    for identifier in audio_predictions:
        if identifier in text_predictions:  # Ensure identifier exists in both sets
            audio_pred = audio_predictions[identifier]
            text_pred = text_predictions[identifier]
            
            if audio_pred == text_pred:
                # Use the prediction when both models agree
                fused_predictions[identifier] = audio_pred
            else:
                # In case of disagreement, average their probabilities to decide the label
                audio_prob = audio_probabilities[identifier]
                text_prob = text_probabilities[identifier]
                avg_prob = (audio_prob + text_prob) / 2
                fused_prediction = 1 if avg_prob > 0.5 else 0
                fused_predictions[identifier] = fused_prediction
    return fused_predictions

def save_fused_predictions(fused_predictions, filename='fused_predictions_mean_rt.txt'):
    with open(filename, 'w') as file:
        for identifier, prediction in fused_predictions.items():
            file.write(f"{identifier},{prediction}\n")
    print(f"Fused predictions saved to {filename}")


def evaluate_predictions(filename='fused_predictions_mean_rt.txt'):
    # Evaluate the predictions in the given file against true labels derived from the identifiers.
    # Assumes that 'PM' or 'PF' in the identifier means depressed (1), and 'CM' or 'CF' means not depressed (0).
    y_true = []
    y_pred = []
    with open(filename, 'r') as file:
        for line in file:
            identifier, prediction = line.strip().split(',')
            true_label = 1 if 'PM' in identifier or 'PF' in identifier else 0
            y_true.append(true_label)
            # Ensure prediction is read as integer
            y_pred.append(int(prediction))
    
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
 

if __name__ == '__main__':
    # Load predictions from both models
    audio_predictions, audio_probabilities = load_predictions_and_probabilities('audio_rt_predictions.txt')
    text_predictions, text_probabilities = load_predictions_and_probabilities('text_rt_predictions.txt')
    
    # Perform decision-level fusion considering disagreements
    fused_predictions = decision_level_fusion(audio_predictions, text_predictions, audio_probabilities, text_probabilities)
    
    # Save the fused predictions
    save_fused_predictions(fused_predictions)
    
    # Evaluate the fused predictions
    evaluate_predictions()
