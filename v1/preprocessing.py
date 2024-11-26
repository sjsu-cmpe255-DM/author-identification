import os
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix



import numpy as np
from scipy.sparse import csr_matrix


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)


# open docs file and read its lines
# Load the dataset
def load_data(directory):
    data = []
    labels = []
    for author in os.listdir(directory):
        author_dir = os.path.join(directory, author)
        if os.path.isdir(author_dir):
            for file_name in os.listdir(author_dir):
                with open(os.path.join(author_dir, file_name), 'r', encoding='utf-8') as file:
                    text = file.read()
                    labels.append(author)
                    data.append(preprocess_text(text))
    return data,labels

def load_labels(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(line.replace("\n", ""))
    return data


def load_sparse_matrix(file_path):
    data, indices, indptr = [], [], [0]
    with open(file_path, 'r') as f:
        for line in f:
            row_data = line.strip().split()
            for i in range(0, len(row_data), 2):
                indices.append(int(row_data[i]))
                data.append(float(row_data[i+1]))
            indptr.append(len(indices))
    return csr_matrix((data, indices, indptr), dtype=float)

def generate_data():
    train_path = "../data/C50train"
    test_path = "../data/C50test"
    train_data,train_labels = load_data(train_path)
    test_data,test_labels = load_data(test_path)


    # Initialize CountVectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the documents into a term-document matrix
    train_X = vectorizer.fit_transform(train_data)
    test_X = vectorizer.transform(test_data)

    # Get the feature names (optional: for mapping indices to words if needed)
    vocabulary = vectorizer.get_feature_names_out()
    # Print the sparse matrix representation for each line

    with open("vocabulary.txt", "w") as file:
        for i,word in enumerate(vocabulary):
            file.write(f"{i} {word}\n")
    # Function to create sparse representation for each row/document
    def get_sparse_representation(row):
        non_zero_indices = row.nonzero()[1]
        return " ".join(f"{idx} {row[0, idx]}" for idx in non_zero_indices)

    # Generate sparse matrix representation for each line
    train_sparse_matrices = [get_sparse_representation(train_X[i]) for i in range(train_X.shape[0])]
    test_sparse_matrices = [get_sparse_representation(test_X[i]) for i in range(test_X.shape[0])]

    # Print the sparse matrix representation for each line
    with open("train_data.txt", "w") as file:
        for sparse_matrix in train_sparse_matrices:
            file.write(f"{sparse_matrix}\n")

    # Print the sparse matrix representation for each line
    with open("test_data.txt", "w") as file:
        for sparse_matrix in test_sparse_matrices:
            file.write(f"{sparse_matrix}\n")



    # Generate a dictionary to map each unique author to a unique ID
    author_to_id = {author: idx for idx, author in enumerate(set(train_labels))}
    train_author_ids = [author_to_id[author] for author in train_labels]
    test_author_ids = [author_to_id[author] for author in test_labels]
    with open("train_labels.txt", "w") as file:
        for author_id in train_author_ids:
            file.write(f"{author_id}\n")

    with open("test_labels.txt", "w") as file:
        for author_id in test_author_ids:
            file.write(f"{author_id}\n")

def data():
    return load_sparse_matrix('./train_data.txt').toarray(),load_sparse_matrix('./test_data.txt').toarray(),load_labels('train_labels.txt'),load_labels('test_labels.txt')