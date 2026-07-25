import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# --------------------------
# 1. Load Dataset
# --------------------------
df = pd.read_csv("Dataset/bmw.csv")

print("Dataset Shape:", df.shape)
print("Sample data:")
print(df.head())

# --------------------------
# 2. Prepare Features & Target
# --------------------------
X = df.drop("price", axis=1)
y = df["price"]

# One-hot encode categorical variables (drop_first=True to avoid multicollinearity)
X_encoded = pd.get_dummies(X, drop_first=True)

print("\nEncoded columns:", X_encoded.columns.tolist())
print("Encoded shape:", X_encoded.shape)

# Save the column names BEFORE scaling (for app.py to reconstruct input)
columns_list = X_encoded.columns.tolist()
joblib.dump(columns_list, "columns.pkl")
print(f"\nSaved {len(columns_list)} columns to columns.pkl")

# --------------------------
# 3. Scale Features
# --------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
print("Saved scaler to scaler.pkl")

# --------------------------
# 4. Train/Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.33, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# --------------------------
# 5. Train Linear Regression Model
# --------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print(f"Intercept: {model.intercept_}")
print(f"Coefficients count: {len(model.coef_)}")

# --------------------------
# 6. Evaluate
# --------------------------
y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
print("\nSample Predictions:")
print(comparison.head(10))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"\nR2 Score: {r2:.4f}")

# --------------------------
# 7. Save the Model
# --------------------------
joblib.dump(model, "LR_BMW.pkl")
print("\nSaved model to LR_BMW.pkl")

print("\n✅ All files saved successfully!")
print("   - LR_BMW.pkl  (trained model)")
print("   - scaler.pkl   (StandardScaler)")
print("   - columns.pkl  (encoded column names)")

