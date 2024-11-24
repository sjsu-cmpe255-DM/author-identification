import os
import pandas as pd
from feature_extraction import extract_tfidf_features
from feature_extraction import apply_pca
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Import models from the models folder
from models.random_forest import train_random_forest
from models.logistic_regression import train_logistic_regression
from models.svm import train_svm



# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('punkt')

# Define stopwords
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Function to load dataset from folder structure
def load_data(base_path):
    data = []
    for dataset_type in ['C50train', 'C50test']:
        dataset_path = os.path.join(base_path, dataset_type)
        for author in os.listdir(dataset_path):
            author_path = os.path.join(dataset_path, author)
            if os.path.isdir(author_path): 
                for file_name in os.listdir(author_path):
                    file_path = os.path.join(author_path, file_name)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                        data.append({'text': text, 'author': author, 'dataset': dataset_type})
    return pd.DataFrame(data)


base_path = '../data'
df = load_data(base_path)
df['cleaned_text'] = df['text'].apply(preprocess_text)


X_tfidf, tfidf_vectorizer = extract_tfidf_features(df['cleaned_text'])
X_reduced, pca = apply_pca(X_tfidf)

y = df['author']

X_train = X_reduced[df['dataset'] == 'C50train']
y_train = y[df['dataset'] == 'C50train']
X_test = X_reduced[df['dataset'] == 'C50test']
y_test = y[df['dataset'] == 'C50test']


model_results = {}


rf_model = train_random_forest(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
model_results['Random Forest'] = rf_accuracy
print("Random Forest Accuracy:", rf_accuracy)
print("Classification Report:\n", classification_report(y_test, rf_pred))


lr_model = train_logistic_regression(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
model_results['Logistic Regression'] = lr_accuracy
print("\nLogistic Regression Accuracy:", lr_accuracy)
print("Classification Report:\n", classification_report(y_test, lr_pred))

svm_model = train_svm(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
model_results['SVM'] = svm_accuracy
print("\nSVM Accuracy:", svm_accuracy)
print("Classification Report:\n", classification_report(y_test, svm_pred))


best_model_name = max(model_results, key=model_results.get)
best_model_accuracy = model_results[best_model_name]
print(f"\nBest Model: {best_model_name} with Accuracy: {best_model_accuracy}")