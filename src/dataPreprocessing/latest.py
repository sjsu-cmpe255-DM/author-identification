import os
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV

import numpy as np

# Download NLTK resources if not already available
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Load the dataset
def load_data(directory):
    data = []
    for author in os.listdir(directory):
        author_dir = os.path.join(directory, author)
        if os.path.isdir(author_dir):
            for file_name in os.listdir(author_dir):
                with open(os.path.join(author_dir, file_name), 'r', encoding='utf-8') as file:
                    text = file.read()
                    data.append({'author': author, 'text': text})
    return pd.DataFrame(data)

# Paths to training and testing sets
train_path = "/Users/pranavtadepu/pyenvs/author-identification/data/C50train"
test_path = "/Users/pranavtadepu/pyenvs/author-identification/data/C50test"

# Load datasets
train_data = load_data(train_path)
test_data = load_data(test_path)

# Apply preprocessing
train_data['cleaned_text'] = train_data['text'].apply(preprocess_text)
test_data['cleaned_text'] = test_data['text'].apply(preprocess_text)

# Encode authors as numerical labels
label_encoder = LabelEncoder()
train_data['author_label'] = label_encoder.fit_transform(train_data['author'])
test_data['author_label'] = label_encoder.transform(test_data['author'])

# Extract features and labels
X_train_text = train_data['cleaned_text']
X_test_text = test_data['cleaned_text']
y_train = train_data['author_label']
y_test = test_data['author_label']

# Transform text data into TF-IDF features with bigrams
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# Parameter grids for hyperparameter tuning
param_grid_lr = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [100, 300, 500]
}

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# RandomizedSearch for Logistic Regression
random_search_lr = RandomizedSearchCV(LogisticRegression(random_state=42),
                                      param_distributions=param_grid_lr,
                                      n_iter=10,
                                      cv=5,
                                      scoring='accuracy',
                                      n_jobs=-1,
                                      random_state=42)
random_search_lr.fit(X_train_tfidf, y_train)
best_lr_model = random_search_lr.best_estimator_

# RandomizedSearch for Random Forest
random_search_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                                      param_distributions=param_grid_rf,
                                      n_iter=10,
                                      cv=5,
                                      scoring='accuracy',
                                      n_jobs=-1,
                                      random_state=42)
random_search_rf.fit(X_train_tfidf, y_train)
best_rf_model = random_search_rf.best_estimator_

# Ensemble Voting Classifier
voting_model = VotingClassifier(estimators=[
    ('lr', best_lr_model),
    ('rf', best_rf_model)
], voting='soft')
voting_model.fit(X_train_tfidf, y_train)

# Logistic Regression Evaluation
y_pred_lr = best_lr_model.predict(X_test_tfidf)
print("Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Random Forest Evaluation
y_pred_rf = best_rf_model.predict(X_test_tfidf)
print("Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Voting Classifier Evaluation
y_pred_voting = voting_model.predict(X_test_tfidf)
print("Voting Classifier Results")
print("Accuracy:", accuracy_score(y_test, y_pred_voting))
print(classification_report(y_test, y_pred_voting))