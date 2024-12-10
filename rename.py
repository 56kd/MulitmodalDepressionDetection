import os

directory = r"C:\Users\44746\Desktop\Androids-Corpus\Androids-Corpus\outputs"

# List all files in the directory
files = os.listdir(directory)

# Iterate through the files and rename if conditions are met
for filename in files:
    if "features.csv" in filename and "text" not in filename:
        new_filename = filename.replace("features.csv", "audio_features.csv")
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        os.rename(old_filepath, new_filepath)
        print(f"Renamed: {filename} -> {new_filename}")
