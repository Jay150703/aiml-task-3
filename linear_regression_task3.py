# Task 3: Linear Regression - House Price Prediction Dataset

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset (using a sample dataset)
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Step 1: Data Preprocessing
print("🔍 Dataset Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# Feature Selection
X = df.drop("medv", axis=1)  # Features
y = df["medv"]               # Target variable

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Fit Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 5: Coefficients Interpretation
print("\n🧮 Coefficients:")
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
print(coeff_df)

# Step 6: Regression Line Plot for one feature (RM vs. medv)
plt.figure(figsize=(8, 5))
sns.regplot(x=df["rm"], y=df["medv"])
plt.title("Regression Line: RM vs MEDV")
plt.xlabel("Average Number of Rooms per Dwelling (RM)")
plt.ylabel("Median Value of Owner-Occupied Homes (MEDV)")
plt.grid(True)
plt.show()
