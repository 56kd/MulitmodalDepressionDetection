import sys
sys.path.append('c:/Users/44746/Desktop/Project')
import unittest
from unittest.mock import patch
from extract_audio import extract_audio_features  

class TestExtractAudioFeatures(unittest.TestCase):

    @patch("os.path.exists")
    @patch("subprocess.run")
    def test_happy_path(self, mock_subprocess, mock_exists):
        mock_exists.return_value = True
        extract_audio_features("existing_file.wav", "config.conf", "output_folder")
        mock_subprocess.assert_called_once()
        
    @patch("os.path.exists")
    @patch("subprocess.run")
    def test_output_csv_file_creation(self, mock_subprocess, mock_exists):
        mock_exists.side_effect = lambda path: True if "existing_file.wav" in path else False
        extract_audio_features("existing_file.wav", "config.conf", "output_folder")
        args, kwargs = mock_subprocess.call_args
        self.assertIn("existing_file_it_audio_features.csv", args[0])


    @patch("os.path.exists")
    @patch("subprocess.run")
    def test_null_audio_file_name(self, mock_subprocess, mock_exists):
        # This test assumes that a NoneType would be passed to the function,
        # demonstrating the function's need for graceful handling of such cases.
        # Python's os.path.exists would throw a TypeError when None is passed as input.
        with self.assertRaises(TypeError):
            extract_audio_features(None, "config.conf", "output_folder")
        mock_subprocess.assert_not_called()

    @patch("os.path.exists")
    @patch("subprocess.run")
    def test_invalid_config_file(self, mock_subprocess, mock_exists):
        mock_exists.return_value = True
        extract_audio_features("audio.wav", "invalid_config.conf", "output_folder")
        args, kwargs = mock_subprocess.call_args
        self.assertIn("invalid_config.conf", args[0])
        # This tests if the function can proceed with an invalid config,
        # subprocess.run should still be called, but the function might need
        # to handle errors from subprocess.run in realistic scenarios.

    @patch("os.path.exists")
    @patch("subprocess.run")
    def test_complex_path(self, mock_subprocess, mock_exists):
        mock_exists.return_value = True
        complex_path = "/path/with special/chars&/audio.wav"
        extract_audio_features(complex_path, "config.conf", "output_folder")
        args, kwargs = mock_subprocess.call_args
        self.assertIn("\"/path/with special/chars&/audio.wav\"", args[0])
        # This test checks how the function handles paths with special characters, ensuring they are appropriately quoted.

if __name__ == "__main__":
    unittest.main()