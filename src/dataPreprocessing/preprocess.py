import os
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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
train_path = "../../data/C50test"
test_path = "../../data/C50test"

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
# Apply the preprocess_text function to the dataset
df['cleaned_text'] = df['text'].apply(preprocess_text)
print(df.head())  # Print the first few rows of the preprocessed dataset
# print(df['cleaned_text'].to_string(index=0))



# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Transform training and testing data
X_train = vectorizer.fit_transform(train_data['text']).toarray()
X_test = vectorizer.transform(test_data['text']).toarray()

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




# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features

# Transform the text data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF matrix shape (training): {X_train_tfidf.shape}")
print(f"TF-IDF matrix shape (testing): {X_test_tfidf.shape}")


# Train a Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

# Predict on test data
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))