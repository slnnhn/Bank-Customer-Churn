# Bank Customer Churn — Model Training Notebook

A Jupyter notebook (Google Colab) that trains multiple machine learning models to predict bank customer churn. This notebook is the model-building backbone for the `headstarter-project1-churn` Streamlit app.

## What It Does

- Loads and explores the bank customer churn dataset (`churn.csv`)
- Preprocesses features: encodes geography/gender, scales numeric columns
- Trains and evaluates multiple classifiers:
  - XGBoost (standard, SMOTE-balanced, and feature-engineered variants)
    - Random Forest
      - Decision Tree
        - K-Nearest Neighbors (KNN)
          - Naive Bayes
            - Support Vector Machine (SVM)
              - Voting Classifier (ensemble)
              - Compares model performance using accuracy, precision, recall, and F1-score
              - Exports trained models as `.pkl` files for use in the Streamlit app

              ## Dataset Features

              Credit score, geography, gender, age, tenure, account balance, number of products, credit card status, active membership, estimated salary, and churn label (Exited).

              ## Tech Stack

              - **Environment:** Google Colab
              - **Libraries:** pandas, scikit-learn, XGBoost, imbalanced-learn (SMOTE), matplotlib, seaborn, pickle

              ## Usage

              1. Open `SN_Headstarter_Project1_Churn.ipynb` in Google Colab or Jupyter
              2. Run all cells in order
              3. Trained model `.pkl` files will be saved for use with the Streamlit app

              ## Related

              See [headstarter-project1-churn](https://github.com/slnnhn/headstarter-project1-churn) for the full Streamlit prediction app.
