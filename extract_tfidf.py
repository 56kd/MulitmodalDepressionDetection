import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Set paths
output_folder = "C:/Users/44746/Desktop/Project/TF-IDF-Interview"
data_dir = "C:/Users/44746/Desktop/Project/Interview"

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def extract_text_features(text_file, output_folder):
    try:
        with open(text_file, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print(f"Text file not found for {text_file}. Skipping.")
        return None
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    features_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    
    output_csv_file = os.path.join(output_folder, f"{os.path.basename(text_file).replace('.txt', '_it_tfidf_features.csv')}")
    features_df.to_csv(output_csv_file, index=False)
    print(f"Extracted text features saved to {output_csv_file}")

def main():
    text_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    for text_file in text_files:
        extract_text_features(text_file, output_folder)

if __name__ == "__main__":
    main()
