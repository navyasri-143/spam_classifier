# **SMS Spam Classifier**  

A machine learning project that classifies SMS messages as **spam** or **ham (not spam)** using **Naive Bayes** and **SVM (Support Vector Machine)**.

## **📌 Overview**  
- Built a **text classification** model trained on the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).  
- Compared two ML algorithms:  
  - **Naive Bayes (MultinomialNB)**  
  - **Support Vector Machine (SVM)**  
- Achieved the best results with **SVM (XX% accuracy)**.  
- Output is displayed in the **terminal** (no web interface).  

## **📂 Dataset**  
- **Source:** [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
- **5,574 SMS messages** (4,827 ham, 747 spam)  
- **Features:**  
  - `label` (spam/ham)  
  - `text` (SMS content)  

## **⚙️ Setup & Installation**  
### **1. Clone the Repository**  
```bash  
git clone https://github.com/your-username/spam-classifier.git  
cd spam-classifier  
```  

### **2. Create & Activate Virtual Environment**  
```bash  
python -m venv venv  
source venv/bin/activate  # Linux/Mac  
.\venv\Scripts\activate   # Windows  
```  

### **3. Install Dependencies**  
```bash  
pip install -r requirements.txt  
```  

### **4. Download NLTK Data (if needed)**  
```python  
import nltk  
nltk.download('punkt')  
nltk.download('stopwords')  
```  

## **🚀 Usage**  
### **1. Training the Model**  
Run the training script:  
```bash  
python train.py  
```  
This will:  
- Preprocess the data (cleaning, tokenization, TF-IDF)  
- Train **Naive Bayes** and **SVM** models  
- Print **accuracy, precision, recall, and F1-score**  

### **2. Making Predictions**  
To classify a new SMS:  
```bash  
python predict.py "Your message here"  
```  
**Example:**  
```bash  
python predict.py "WIN a free iPhone! Click now!"  
```  
**Output:**  
```  
Prediction: SPAM  
Confidence: 98.5%  
```  

## **📊 Results**
training results
=== Model Evaluation ===​

MultinomialNB Results:​

Accuracy: 96.41%​

Classification Report:​

              precision    recall  f1-score   support​

​

           0       0.96      1.00      0.98       966​

           1       0.99      0.74      0.85       149​

​

    accuracy                           0.96      1115​

   macro avg       0.98      0.87      0.91      1115​

weighted avg       0.97      0.96      0.96      1115​

SVC Results:​

Accuracy: 98.65%​

Classification Report:​

   precision    recall  f1-score   support​

​

           0       0.98      1.00      0.99       966​

           1       1.00      0.90      0.95       149​

    accuracy                           0.99      1115​

   macro avg       0.99      0.95      0.97      1115​

weighted avg       0.99      0.99      0.99      1115​


prediction:
​

Spam Classifier Interactive Mode (type 'quit' to exit)​

​

Enter a message to classify: fee cash$300​

Result: SPAM (confidence: 50.94%)​

Details: Not Spam: 49.06% | Spam: 50.94%​

Enter a message to classify:  LIMITED TIME OFFER! 50% discount today only!​

Result: NOT SPAM (confidence: 73.70%)​

Details: Not Spam: 73.70% | Spam: 26.30%​

Enter a message to classify: ​

Result: NOT SPAM (confidence: 86.58%)​

Details: Not Spam: 86.58% | Spam: 13.42%​

Enter a message to classify: Please review the document I sent you​

Result: NOT SPAM (confidence: 83.66%)​

Details: Not Spam: 83.66% | Spam: 16.34%​

​



## **📂 Project Structure**  
```  
spam-classifier/  
├── data/  
│   └── spam.csv               # Dataset  
├── models/  
│   ├── naive_bayes_model.pkl  # Trained Naive Bayes  
│   └── svm_model.pkl          # Trained SVM  
├── src/  
│   ├── preprocess.py          # Text cleaning & TF-IDF  
│   ├── train.py               # Model training  
│   ├── predict.py             # Prediction script  
│   └── utils.py               # Helper functions  
├── requirements.txt           # Dependencies  
└── README.md                  # This file  
```  

## **🔧 Future Improvements**  
- Try **deep learning models (LSTM, BERT)**  
- Add **Flask/Django API** for web integration  
- Deploy as a **Chrome extension** for real-time spam detection  
