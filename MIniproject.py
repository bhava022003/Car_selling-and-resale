# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 2. Load dataset
df = pd.read_csv("car data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# 3. Data preprocessing
df.columns = df.columns.str.strip()

# Remove duplicate records
df = df.drop_duplicates()

# Remove rows with invalid original prices
df = df[df["Present_Price"] > 0].copy()

# Create resale classification target
df["Resale_Ratio"] = (
    df["Selling_Price"] / df["Present_Price"]
)

df["Resale_Status"] = np.where(
    df["Resale_Ratio"] >= 0.60,
    1,
    0
)

# 4. Select input features
features = [
    "Year",
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner"
]

X = df[features]

# Linear Regression target
y_price = df["Selling_Price"]

# Logistic Regression target
y_resale = df["Resale_Status"]

# 5. Identify column types
numeric_features = [
    "Year",
    "Present_Price",
    "Kms_Driven",
    "Owner"
]

categorical_features = [
    "Fuel_Type",
    "Seller_Type",
    "Transmission"
]

# 6. Preprocessing pipeline
def create_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    return preprocessor

# 7. Split dataset
X_train, X_test, y_price_train, y_price_test, y_resale_train, y_resale_test = train_test_split(
    X,
    y_price,
    y_resale,
    test_size=0.2,
    random_state=42,
    stratify=y_resale
)

# ==========================================
# MODEL 1: LINEAR REGRESSION
# ==========================================

print("\n===== LINEAR REGRESSION =====")

linear_model = Pipeline([
    ("preprocessing", create_preprocessor()),
    ("model", LinearRegression())
])

# Train model
linear_model.fit(X_train, y_price_train)

# Predict selling prices
price_prediction = linear_model.predict(X_test)

# Evaluate model
print("MAE:", mean_absolute_error(
    y_price_test, price_prediction
))

print("MSE:", mean_squared_error(
    y_price_test, price_prediction
))

print("RMSE:", np.sqrt(mean_squared_error(
    y_price_test, price_prediction
)))

print("R2 Score:", r2_score(
    y_price_test, price_prediction
))

# Actual vs Predicted graph
plt.figure(figsize=(7, 5))

plt.scatter(y_price_test, price_prediction)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")

plt.show()

# ==========================================
# MODEL 2: LOGISTIC REGRESSION
# ==========================================

print("\n===== LOGISTIC REGRESSION =====")

logistic_model = Pipeline([
    ("preprocessing", create_preprocessor()),
    ("model", LogisticRegression(max_iter=1000))
])

# Train model
logistic_model.fit(X_train, y_resale_train)

# Predict resale status
resale_prediction = logistic_model.predict(X_test)

# Evaluate model
print("Accuracy:",
      accuracy_score(y_resale_test, resale_prediction))

print("\nClassification Report:")
print(classification_report(
    y_resale_test,
    resale_prediction,
    target_names=["Low Resale", "Good Resale"],
    zero_division=0
))

# Confusion Matrix
cm = confusion_matrix(
    y_resale_test,
    resale_prediction
)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low Resale", "Good Resale"],
    yticklabels=["Low Resale", "Good Resale"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Resale Classification Confusion Matrix")

plt.show()

# ==========================================
# PREDICT A NEW CAR (USER INPUT)
# ==========================================

print("\n" + "=" * 45)
print("        ENTER CAR DETAILS FOR PREDICTION")
print("=" * 45)

try:
    year = int(input("Enter Manufacturing Year (e.g., 2018): "))
    present_price = float(input("Enter Present Showroom Price in Lakhs (e.g., 8.5): "))
    kms_driven = int(input("Enter Kilometers Driven (e.g., 45000): "))

    fuel_input = input("Enter Fuel Type (Petrol / Diesel / CNG): ").strip()
    fuel_type = "CNG" if fuel_input.upper() == "CNG" else fuel_input.capitalize()

    seller_type = input("Enter Seller Type (Dealer / Individual): ").strip().capitalize()
    transmission = input("Enter Transmission (Manual / Automatic): ").strip().capitalize()
    owner = int(input("Enter Number of Previous Owners (0, 1, 2, 3): "))

    new_car = pd.DataFrame([{
        "Year": year,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission,
        "Owner": owner
    }])

    # Linear Regression prediction
    predicted_price = linear_model.predict(new_car)[0]

    # Logistic Regression prediction
    predicted_resale = logistic_model.predict(new_car)[0]

    print("\n" + "=" * 45)
    print("              PREDICTION RESULTS")
    print("=" * 45)
    print(f"Predicted Selling Price : {round(predicted_price, 2)} Lakhs")

    if predicted_resale == 1:
        print("Resale Classification   : GOOD RESALE (≥ 60% value retained)")
    else:
        print("Resale Classification   : LOW RESALE (< 60% value retained)")
    print("=" * 45)

except Exception as e:
    print("\nError taking input or predicting:", e)
