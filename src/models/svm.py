from sklearn.svm import SVC

def train_svm(X_train, y_train):
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    return model