**Overview**

This project focuses creating a multimodal depression detection model from audio and text data using machine learning and deep learning techniques. It utilises Python libraries such as Pandas, NumPy, PyTorch, Transformers, NLTK, and sklearn, and integrates OpenSMILE for audio feature extraction. The project is structured around five main components:


1. Audio Feature Extraction (extract_audio.py): Extracts audio features using OpenSMILE.
2. Text Sentiment Classification (BERTC.py): Utilizes a pre-trained BERT model for sentiment analysis on text data.
3. Hybrid CNN-LSTM Model for Audio Emotion Recognition (HybridCNNLSTM.py): Implements a deep learning model combining Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks for emotion recognition from audio data.
4. Decision level fusion to integrate predictions from both audio and text analysis models to improve overall prediction accuracy.
5. Feature correlation to investigate the interplay between read and spontaneous speech in depression detection.

**Data Files**
This project utilizes several data files for processing and analysis:

- Androids.conf: OpenSMILE configuration file used for extracting audio features.
- InterviewFolds.csv & ReadingFolds.csv: CSV files containing fold information for structured cross-validation in audio and text data processing.
- Audio and text data files located in specified directories, structured according to the project's needs.


**Directories**

Interview Task Directories:
- _Interview_: contains all audio and text files for interview task.
- _Audio-Interview_: .csv files containing features extracted using openSMILE for each .wav file in Interview Directory.

Reading Task Directories:
- _Reading_: contains all audio and text files for reading task.
- _Audio-Reading_: .csv files containing features extracted using openSMILE for each .wav file in _Reading_ Directory.

AudioModels:
- Contains all machine learning models for depression classification on audio features.

TextModels:
- Contains all machine learning models for depression classification on text features.

DecisionLevelFusion:
- Contains a variety of decision-level fusion algorithms to integrate predictions from both audio and text analysis.

FeatureLevelFusion:
- Machine learning models that combine data from both text and audio sources before they are fed into a learning model.  

Interplay:
- _heatmap.py_: Heatmap generation for feature correlation analysis.
- _audio_featuresx.csv_ : .csv file containing features from all audio files.

**Requirements**

### Hardware Requirements
- **CPU**: Intel i5 or higher recommended for efficient computation.
- **RAM**: Minimum 8 GB; 16 GB recommended for handling large datasets.
- **Storage**: At least 10 GB of free disk space for storing datasets, model checkpoints, and outputs.

### Software Requirements
- **Operating System**: Windows 10, macOS, or Linux.

- **Python Environment**: Python 3.11 recommended. Use of a virtual environment (`venv` or `conda`) is advised to manage dependencies.

To run this project, you must have the following libraries installed:

- OpenSMILE
- Pandas
- NumPy
- PyTorch
- Collections
- os
- sys
- Transformers library
- NLTK
- sklearn
- Seaborn
- Matplotlib
- Math
- Datasets
- unittest
- pytest

Python dependencies can be installed via pip.

Ensure OpenSMILE is properly installed and configured on your system as per the instructions on the [openSMILE website](https://www.audeering.com/research/opensmile/) .


**Installation**

### Installation Instructions
1. **Clone the repository:**
   ```bash
   git clone https://gitlab.eeecs.qub.ac.uk/40231743/csc4006-preliminary-code.git
   cd depression-detection
   ```

2. **Set up a Python virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install pandas numpy torch torchvision torchaudio transformers nltk scikit-learn datasets seaborn matplotlib pytest
   ```

4. **Verify Installation:**
   - Run a simple script to check library imports and basic functionality:
     ```python
     import pandas as pd
     import torch
     print(f"Installed pandas version: {pd.__version__}")
     print(f"PyTorch is cuda available: {torch.cuda.is_available()}")
     ```

   - Execute the script:
     ```bash
     python test_install.py
     ```

   - **Expected Output:**
     ```
     Installed pandas version: 1.3.4
     PyTorch is cuda available: True/False
     ```
   This confirms that the necessary libraries are installed and PyTorch's GPU support is correctly configured according to your system's capabilities.



**Configuration**
Before executing the scripts, configure the paths in the Python files to match your system setup, including the paths to OpenSMILE binaries, configuration files, and data directories:

opensmile_bin_path: Path to the OpenSMILE binaries.  
config_file_path: Path to the OpenSMILE configuration file.  
smil_extract_path: Path to the SMILExtract executable from OpenSMILE.  
output_folder: Path where the output from OpenSMILE will be stored.  
data_dir: Directory containing the data to be processed.  
fold_csv_path: CSV file containing the pre-defined cross validation folds (ReadingFolds.csv or InterviewFolds.csv).


Replace the placeholders with the correct paths on your system.  

**Usage**

The project involves several steps, starting from audio processing with OpenSMILE, data manipulation with Pandas and NumPy, to NLP and machine learning tasks with PyTorch and the Transformers library.

1. extract_audio.py
Extracts audio features from .wav files using OpenSMILE. It requires specifying paths to the OpenSMILE binary, configuration file, input audio files, and output directory for the extracted features.

    extract_bert.py / extract_tfidf.py - extract text features from .txt files (necessary for text models excluding BERT.py)

2. TextModels/BERT.py
Performs sentiment analysis on Italian text data using a pre-trained BERT model (neuraly/bert-base-italian-cased-sentiment). It processes text files, tokenizes and groups sentences, and uses the BERT model for classification. Outputs include accuracy, precision, recall, F1 score. Predictions with probabilities are saved to a txt file "text_xt_predictions.txt", where x in xt is r for reading task and i for interview task.

3. AudioModels/HybridCNNLSTM.py
Implements a hybrid CNN-LSTM model for emotion recognition from audio features. The script includes data loading, preprocessing (normalization), model definition, training, evaluation, and prediction aggregation through majority voting. Outputs include accuracy, precision, recall, F1 score. Predictions with probabilities are saved to a txt file "audio_xt_predictions.txt", where x in xt is r for reading task and i for interview task.

4. WeightedAverage.py
Integrates predictions from both audio and text analysis models through a decision-level fusion algorithm. It compares predictions and probabilities from both models to produce a final, fused prediction, aiming to improve overall prediction accuracy. The fusion process involves decision-making based on agreement between models or using weighted averages of probabilities in case of disagreement.

5. Interplay/heatmap.py
Generates feature correlation matrices for 'Read Depressed', 'Read Non-Depressed', 'Spontaneous Depressed' and 'Spontaneous Non-Depressed' subgroups, to aid the investigation of the interplay between read and spontaneous speech in depression detection.


**Replication Guide**

1. Audio Feature Extraction:
Unzip Audio-Interview.zip and Audio-Reading.zip - these directories contain the extracted audio features in .csv format.

(For a true replication, download the [Androids Corpus](https://github.com/androidscorpus/data). Combine each patients segmented interview .wav files into one. Download [Buzz](https://buzzcaptions.com/) to generate transcriptions. Run extract_audio.py.)

2. Text Sentiment Classification:
Unzip ReadingTranscriptions. zip and InterviewTranscriptions.zip - these contain the transcription .txt files. Ensure the data paths in TextModels/BERT.py point to the transcriptions and run the script for sentiment analysis on text data. Run the output file in Metrics.py for all performance metrics.

3. Audio Emotion Recognition:
Configure paths in AudioModels/HybridCNNLSTM.py. Run the script to train and evaluate the hybrid CNN-LSTM model on audio data. Run the output file in Metrics.py for all performance metrics.

4. Decision Level Fusion:  
Configure paths in/DecisionLevelFusion/WeightedAverage.py. Run the script to perform decision-level fusion of predictions from the audio and text models. This script will load predictions, perform fusion, save the fused predictions, and evaluate the performance. Run the output file in Metrics.py for all performance metrics.

5. Feature Correlation:
Unzip audio_featuresx.csv. Configure paths in Interplay/heatmap.py. Run the script and the feature correlations will display on the screen and save to the Interplay directory.
