# ==========================================
# ANIMAL DANGER PREDICTION
# USING LOGISTIC REGRESSION
# ==========================================

# STEP 1: Import libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)

# STEP 2: Load dataset

df = pd.read_csv("data.csv")

print("Original Dataset:")
print(df.head())

# STEP 3: Check dataset information

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# STEP 4: Clean column names

df.columns = df.columns.str.strip()

# STEP 5: Remove unwanted spaces and lowercase text

for col in df.columns:
    df[col] = df[col].apply(
        lambda x: x.strip().lower()
        if isinstance(x, str) else x
    )

# STEP 6: Remove duplicate rows

df = df.drop_duplicates()

# STEP 7: Remove missing target values

df = df.dropna(subset=["Dangerous"])

# STEP 8: Separate independent and dependent variables

X = df.drop("Dangerous", axis=1)
y = df["Dangerous"]

print("\nIndependent Variables:")
print(X.head())

print("\nDependent Variable:")
print(y.head())

# STEP 9: Identify categorical columns

cat_cols = X.select_dtypes(include="object").columns

# STEP 10: Create preprocessing pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

# STEP 11: Split dataset into training and testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# STEP 12: Create Logistic Regression model

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

# STEP 13: Train the model

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# STEP 14: Predict testing data

y_pred = model.predict(X_test)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

# STEP 15: Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy Score:", accuracy * 100, "%")

# STEP 16: Balanced accuracy

print(
    "Balanced Accuracy:",
    balanced_accuracy_score(y_test, y_pred) * 100,
    "%"
)

# STEP 17: Confusion Matrix

print("\nConfusion Matrix:")

print(confusion_matrix(
    y_test,
    y_pred,
    labels=["no", "yes"]
))

# STEP 18: Classification Report

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    labels=["no", "yes"],
    zero_division=0
))

# STEP 19: Predict a new animal

new_animal = pd.DataFrame([{
    "AnimalName": "Dog",
    "symptoms1": "Fever",
    "symptoms2": "Diarrhea",
    "symptoms3": "Vomiting",
    "symptoms4": "Weight loss",
    "symptoms5": "Dehydration"
}])

prediction = model.predict(new_animal)

print("\nNew Animal Prediction:")

if prediction[0] == "yes":
    print("Animal is predicted as DANGEROUS")
else:
    print("Animal is predicted as NOT DANGEROUS")

print("\nProject Completed Successfully!")