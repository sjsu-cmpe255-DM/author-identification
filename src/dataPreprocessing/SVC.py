import os
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Download NLTK data if needed
nltk.download('stopwords')

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

train_data = load_data(train_path)
test_data = load_data(test_path)

# Apply preprocessing
train_data['cleaned_text'] = train_data['text'].apply(preprocess_text)
test_data['cleaned_text'] = test_data['text'].apply(preprocess_text)

# Encode the authors as numerical labels
label_encoder = LabelEncoder()
train_data['author_label'] = label_encoder.fit_transform(train_data['author'])
test_data['author_label'] = label_encoder.transform(test_data['author'])

# Transform the text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(train_data['cleaned_text'])
X_test_tfidf = vectorizer.transform(test_data['cleaned_text'])
y_train = train_data['author_label']
y_test = test_data['author_label']

print(f"TF-IDF matrix shape (training): {X_train_tfidf.shape}")
print(f"TF-IDF matrix shape (testing): {X_test_tfidf.shape}")

# Define the parameter grid for SVC
param_grid_svc = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'kernel': ['linear', 'rbf'],  # Kernel types
    'gamma': ['scale', 'auto']  # Kernel coefficient for 'rbf'
}

# Perform Grid Search with Cross-Validation for SVC
print("Training Support Vector Classifier with Grid Search...")
grid_svc = GridSearchCV(SVC(random_state=42), param_grid_svc, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_svc.fit(X_train_tfidf, y_train)

# Best parameters and model
print("Best Parameters for SVC:", grid_svc.best_params_)
best_svc_model = grid_svc.best_estimator_

# Predict using the best SVC model
y_pred_svc = best_svc_model.predict(X_test_tfidf)

# Evaluate the SVC model
print("Support Vector Classifier Results")
print("Accuracy:", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))

# Random Forest for comparison (optional)
print("Training Random Forest for comparison...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_test_tfidf)

# Evaluate the Random Forest model
print("Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))