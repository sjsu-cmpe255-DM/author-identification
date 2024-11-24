from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

def extract_tfidf_features(text_data, max_features=5000):

    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_features = tfidf_vectorizer.fit_transform(text_data).toarray()
    return tfidf_features, tfidf_vectorizer


def apply_pca(features, n_components=100):

    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features, pca