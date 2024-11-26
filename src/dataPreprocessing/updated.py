import os
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from Config import apply_dimensionality_reduction

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

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
# print(train_data.shape)
# print(train_data.loc[51])
# print(train_data.head())
# print(train_data.info())


# Apply preprocessing
# Example usage

text = "This is a test sentence for preprocessing."
cleaned_text = preprocess_text(text)
print(cleaned_text)

df = pd.DataFrame(train_data)
dftest = pd.DataFrame(test_data)
# Apply the preprocess_text function to the dataset
df['cleaned_text'] = df['text'].apply(preprocess_text)
dftest['cleaned_text'] = dftest['text'].apply(preprocess_text)
print(df.head())  # Print the first few rows of the preprocessed dataset
print(dftest.head())


# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
# Transform training and testing data
X_train = vectorizer.fit_transform(df['cleaned_text']).toarray()
X_test = vectorizer.transform(dftest['cleaned_text']).toarray()

print(f"X_train head printing: {X_train.shape}")
pcadata,svd = apply_dimensionality_reduction(X_train)
pcadata_test = svd.transform(X_test)  # Use the same SVD object to transform test data

print("Completed Dimensional Reduction")
print(pcadata.shape)
dfDR_train = pd.DataFrame(pcadata)
dfDR_test = pd.DataFrame(pcadata_test)
print(dfDR_train.head())
y_train = train_data['author']
y_test = test_data['author']

print(f"Training Data Shape: {X_train.shape}")



# Encode the authors as numerical labels
label_encoder = LabelEncoder()
df['author_label'] = label_encoder.fit_transform(df['author'])

# Check the encoded labels
print(df[['author', 'author_label']].head())



# Features and target
X = df['cleaned_text']  # Preprocessed text
y = df['author_label']  # Encoded author labels

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")



# # Initialize the TF-IDF Vectorizer
# vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features
# # Transform the text data
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# print(f"TF-IDF matrix shape (training): {X_train_tfidf.shape}")
# print(f"TF-IDF matrix shape (testing): {X_test_tfidf.shape}")


# Train a Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(pcadata, y_train)


RFmodel = RandomForestClassifier(n_estimators=100, random_state=42)
RFmodel.fit(pcadata, y_train)


# Predict on test data
y_pred = model.predict(pcadata_test)
yrf_pred = RFmodel.predict(pcadata_test)
# Evaluate the model
print("AccuracyLR:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("AccuracyRF:", accuracy_score(y_test, yrf_pred))
print(classification_report(y_test, yrf_pred))