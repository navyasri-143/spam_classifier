import pandas as pd
import os
from pathlib import Path
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')

def preprocess_text(text: str) -> str:
    """Preprocess text by:
    1. Converting to lowercase
    2. Removing punctuation (but keeping numbers and $)
    3. Removing stopwords (but keeping important words)
    4. Stemming words (but not numbers)
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except $ and %
    text = ''.join([char for char in text if char not in string.punctuation.replace('$', '')])
    
    # Tokenize
    words = text.split()
    
    # Create custom stopwords list that keeps important words
    base_stopwords = set(stopwords.words('english'))
    # Keep these words even if they're normally stopwords
    keep_words = {'now', 'not', 'no', 'nor'}
    custom_stopwords = base_stopwords - keep_words
    
    # Remove stopwords (but keep numbers and our special words)
    words = [
        word 
        for word in words 
        if (word not in custom_stopwords) or 
           any(c.isdigit() for c in word) or
           (word in keep_words)
    ]
    
    # Stemming (don't stem numbers or our special words)
    stemmer = PorterStemmer()
    words = [
        stemmer.stem(word) 
        if not (any(c.isdigit() for c in word) or (word in keep_words))
        else word
        for word in words
    ]
    
    return ' '.join(words)

def load_and_preprocess_data(filepath=None):
    """Load and preprocess the dataset"""
    # Use default path if none provided
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), '../data/spam.csv')
    
    try:
        # Verify file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found at {filepath}\nCurrent directory: {os.getcwd()}")
        
        df = pd.read_csv(filepath, encoding='latin-1')
        df = df[['v1', 'v2']]  # Select only label and text columns
        df.columns = ['label', 'message']  # Rename columns
        
        # Preprocess messages
        df['processed_message'] = df['message'].apply(preprocess_text)
        
        # Convert labels to binary
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

__all__ = ['preprocess_text', 'load_and_preprocess_data']