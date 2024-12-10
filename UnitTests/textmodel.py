import os
import unittest
from unittest.mock import patch
from nltk.tokenize.punkt import PunktSentenceTokenizer
from TextModels.BERT import group_sentences, save_predictions_to_file
class TestGroupSentences(unittest.TestCase):
    
    def test_empty_input(self):
        self.assertEqual(group_sentences(""), [])
    
    def test_null_input(self):
        with self.assertRaises(TypeError):
            group_sentences(None)
    
    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            group_sentences(123)  
    
    def test_single_sentence(self):
        text = "Questa è una frase."
        self.assertEqual(group_sentences(text), [text])
    
    def test_two_sentences(self):
        text = "Questa è la prima frase. Questa è la seconda frase."
        self.assertEqual(group_sentences(text), [text])
    
    def test_odd_number_of_sentences(self):
        text = "Questa è la prima frase. Questa è la seconda frase. Questa è la terza frase."
        expected = ["Questa è la prima frase. Questa è la seconda frase.", "Questa è la terza frase."]
        self.assertEqual(group_sentences(text), expected)
    
    def test_even_number_of_sentences(self):
        text = "Uno. Due. Tre. Quattro."
        expected = ["Uno. Due.", "Tre. Quattro."]
        self.assertEqual(group_sentences(text), expected)
    
    def test_large_input(self):
        text = " ".join(["Frase."] * 1000) # 1000 sentences.
        result = group_sentences(text)
        self.assertEqual(len(result), 500)  # Should group into 500 pairs of sentences.
    
    def test_input_with_newlines(self):
        text = "Primo.\n\nSecondo.\nTerzo.\n\nQuarto."
        expected = ["Primo. Secondo.", "Terzo. Quarto."]
        self.assertEqual(group_sentences(text), expected)
    
    def test_language_other_than_italian(self):
        # Assuming the default language is Italian, and an English text is an unexpected case.
        text = "This is the first sentence. This is the second sentence."
        expected = ["This is the first sentence. This is the second sentence."] 
        self.assertEqual(group_sentences(text), expected)


if __name__ == '__main__':
    unittest.main()
