# Author Identification Using Text Classification

## **Project Description**
This project aims to identify the author of a given news article based on their unique linguistic style. By leveraging machine learning techniques and natural language processing (NLP), we analyze stylistic elements such as word choice, syntax, and patterns to classify authorship.

---

## **Dataset**
We use the **Reuter_50_50** dataset:
- 2,500 news articles written by 50 authors (50 articles per author).
- Split into:
  - **Training Set**: 1,250 articles (`C50train`)
  - **Testing Set**: 1,250 articles (`C50test`)

---

## **Goal**
The primary goal is to test multiple machine learning models and determine which one performs best in identifying authorship.

### Models Implemented:
1. Random Forest Classifier (`random_forest.py`)
2. Logistic Regression (`logistic_regression.py`)
3. Support Vector Machine (SVM) (`svm.py`)
4. XGBoost Classifier (`xgboost_model.py`)

---

## **How to Run the Project**

### Prerequisites:
- Python 3.8 or above.
- Install dependencies using `pip`:

```bash
pip install -r requirements.txt