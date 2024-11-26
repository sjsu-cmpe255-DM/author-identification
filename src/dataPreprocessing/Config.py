from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np



# def extract_tfidf_features(text_data, max_features=5000):

#     tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
#     tfidf_features = tfidf_vectorizer.fit_transform(text_data).toarray()
#     return tfidf_features, tfidf_vectorizer



#with pca got 2080 with SVD got 1500 around
def apply_dimensionality_reduction(features, variance_threshold=0.95):
    # Initialize TruncatedSVD without specifying n_components first
    svd = TruncatedSVD(n_components=min(features.shape)-1, random_state=42)
    svd.fit(features)
    
    # Calculate cumulative variance ratio
    cumsum = np.cumsum(svd.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= variance_threshold) + 1
    print(f"Components: {n_components}")
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumsum) + 1), cumsum, 'b-')
    plt.axhline(y=variance_threshold, color='r', linestyle='--')
    plt.axvline(x=n_components, color='g', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance vs. Number of Components')
    plt.grid(True)
    plt.show()
    
    # Apply TruncatedSVD with optimal components
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42,
        algorithm='randomized',
        n_iter=5
    )
    reduced_features = svd.fit_transform(features)
    
    print(f"Selected {n_components} components explaining {variance_threshold*100:.1f}% of variance")
    return reduced_features, svd
