import joblib
import os
from pathlib import Path
from .preprocess import preprocess_text  # Only import what's available

class SpamClassifier:
    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        # Set default paths if none provided
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '../models/nb_spam_model.pkl')
        if vectorizer_path is None:
            vectorizer_path = os.path.join(os.path.dirname(__file__), '../models/tfidf_vectorizer.pkl')
        
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not load model files: {e}")

    def predict(self, message: str) -> dict:
        """Predict if a message is spam"""
        processed = preprocess_text(message)
        vectorized = self.vectorizer.transform([processed])
        
        prediction = self.model.predict(vectorized)[0]
        probabilities = self.model.predict_proba(vectorized)[0]
        
        return {
            'is_spam': bool(prediction),
            'confidence': float(probabilities[prediction]),
            'probabilities': {
                'not_spam': float(probabilities[0]),
                'spam': float(probabilities[1])
            }
        }

def interactive_predict():
    """Interactive prediction mode"""
    try:
        classifier = SpamClassifier()
        print("Spam Classifier Interactive Mode (type 'quit' to exit)")
        
        while True:
            message = input("\nEnter a message to classify: ")
            if message.lower() in ('quit', 'exit', 'q'):
                break
            
            try:
                result = classifier.predict(message)
                status = "SPAM" if result['is_spam'] else "NOT SPAM"
                print(f"\nResult: {status} (confidence: {result['confidence']:.2%})")
                print(f"Details: Not Spam: {result['probabilities']['not_spam']:.2%} | "
                      f"Spam: {result['probabilities']['spam']:.2%}")
            except Exception as e:
                print(f"Error processing message: {e}")
    
    except Exception as e:
        print(f"Failed to initialize classifier: {e}")

if __name__ == "__main__":
    interactive_predict()