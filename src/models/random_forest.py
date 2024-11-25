from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Parameter distributions for RandomizedSearchCV
param_grid = {
    'n_estimators': randint(100, 1000),
    'max_depth': [None] + list(randint(3, 50).rvs(10)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
    )
    # Fit the random search
    rf_random.fit(X_train, y_train)
    # Get best parameters
    # best_params = rf_random.best_params_
    return rf_random