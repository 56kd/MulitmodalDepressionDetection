import os
import subprocess
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Set paths
opensmile_bin_path = "C:/Users/44746/opensmile/bin"
config_file_path = "C:/Users/44746/Desktop/Androids-Corpus/Androids-Corpus/Androids.conf"
smil_extract_path = "C:/Users/44746/opensmile/bin/SMILExtract.exe"
output_folder = "C:/Users/44746/Desktop/Androids-Corpus/Androids-Corpus/outputs"
data_dir = "C:/Users/44746/Desktop/Androids-Corpus/Androids-Corpus/audio/ALL"

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def extract_audio_features(audio_file, config_file, output_folder):
    if not os.path.exists(audio_file):
        print(f"Audio file not found for {audio_file}. Skipping.")
        return None

    output_csv_file = os.path.join(output_folder, f"{os.path.basename(audio_file).replace('.wav', '_audio_features.csv')}")
    command = f"{smil_extract_path} -C {config_file} -I \"{audio_file}\" -O \"{output_csv_file}\""
    subprocess.run(command, shell=True)
    print(f"Extracted audio features saved to {output_csv_file}")

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
    
    output_csv_file = os.path.join(output_folder, f"{os.path.basename(text_file).replace('.txt', '_text_features.csv')}")
    features_df.to_csv(output_csv_file, index=False)
    print(f"Extracted text features saved to {output_csv_file}")

def main():
    audio_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
    text_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]

    for audio_file in audio_files:
        extract_audio_features(audio_file, config_file_path, output_folder)
    
    for text_file in text_files:
        extract_text_features(text_file, output_folder)

if __name__ == "__main__":
    main()
