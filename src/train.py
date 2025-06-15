from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib
import os
from pathlib import Path
from .preprocess import load_and_preprocess_data

def train_models(data_path=None):
    """Train and save spam classification models"""
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Load and preprocess data
    df = load_and_preprocess_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_message'], 
        df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    # Feature extraction
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Save vectorizer with absolute path
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    joblib.dump(tfidf, vectorizer_path)
    
    # Train Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_path = os.path.join(models_dir, 'nb_spam_model.pkl')
    joblib.dump(nb_model, nb_path)
    
    # Train SVM
    svm_model = SVC(kernel='linear', C=1, probability=True)
    svm_model.fit(X_train_tfidf, y_train)
    svm_path = os.path.join(models_dir, 'svm_spam_model.pkl')
    joblib.dump(svm_model, svm_path)
    
    # Evaluate models
    def evaluate(model, X, y):
        y_pred = model.predict(X)
        print(f"\n{model.__class__.__name__} Results:")
        print(f"Accuracy: {accuracy_score(y, y_pred):.2%}")
        print("Classification Report:")
        print(classification_report(y, y_pred))
    
    print("=== Model Evaluation ===")
    evaluate(nb_model, X_test_tfidf, y_test)
    evaluate(svm_model, X_test_tfidf, y_test)
    
    print(f"\nModels saved to: {models_dir}")

if __name__ == "__main__":
    # Use absolute path when running directly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data_path = os.path.join(project_root, 'data', 'spam.csv')
    train_models(default_data_path)