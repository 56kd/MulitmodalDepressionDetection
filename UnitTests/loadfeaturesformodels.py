import unittest
import os
import sys

sys.path.append('c:/Users/44746/Desktop/Project')
from AudioModels.HybridCNNLSTM import load_features_for_fold

def setUp(self):
    """Set up test environment, including valid files."""
    self.test_directory = "test_data"
    os.makedirs(self.test_directory, exist_ok=True)
    self.test_filenames = ['test_file_1', 'test_file_2']

    # Adjusted to generate enough data for the expected number of segments
    data_rows = 448  # Ensuring enough rows for 6 segments with the given L and step
    for filename in self.test_filenames:
        path = os.path.join(self.test_directory, f"{filename}_rt_audio_features.csv")
        with open(path, 'w') as f:
            f.write("feature1,feature2,feature3,feature4,feature5\n")
            for _ in range(data_rows):  # Adjusted number of rows
                f.write("1,2,3,4,5\n")

        # Creating a malformed file
        malformed_path = os.path.join(self.test_directory, "malformed_file_rt_audio_features.csv")
        with open(malformed_path, 'w') as f:
            f.write("This, is, not, proper, CSV\n1, 2, 3\na, b, c")

    def tearDown(self):
        """Clean up after tests."""
        for root, dirs, files in os.walk(self.test_directory, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
        os.rmdir(self.test_directory)

    def test_empty_input(self):
        """Test handling of empty input."""
        segments, labels, identifiers = load_features_for_fold(self.test_directory, [])
        self.assertEqual(len(segments), 0)
        self.assertEqual(len(labels), 0)
        self.assertEqual(len(identifiers), 0)

    def test_happy_case(self):
        """Test normal case with valid input."""
        segments, labels, identifiers = load_features_for_fold(self.test_directory, self.test_filenames)
        self.assertEqual(len(segments), 6, "Expected 6 segments for valid input")


    def test_invalid_input_paths(self):
        """Test with non-existent files."""
        segments, labels, identifiers = load_features_for_fold(self.test_directory, ['invalid_file_1', 'invalid_file_2'])
        self.assertEqual(len(segments), 0, "Segments should be empty for non-existent files")
        self.assertEqual(len(labels), 0, "Labels should be empty for non-existent files")
        self.assertEqual(len(identifiers), 0, "Identifiers should be empty for non-existent files")

    def test_malformed_data(self):
        """Test handling of malformed data file."""
        segments, labels, identifiers = load_features_for_fold(self.test_directory, ['malformed_file'])
        self.assertEqual(len(segments), 0, "Segments should be empty for malformed files")

if __name__ == "__main__":
    unittest.main()
