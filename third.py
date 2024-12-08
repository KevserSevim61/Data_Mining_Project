# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data = pd.read_csv("auto-mpg.csv")

# Convert 'horsepower' to numeric and handle missing values
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data['horsepower'].fillna(data['horsepower'].median(), inplace=True)

# Drop the 'car name' column as it is irrelevant for regression
data.drop(columns=['car name'], inplace=True)

# Separate features and target variable
X = data.drop(columns=['mpg'])
y = data['mpg']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 2: Define and train models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(random_state=42)
}

results = {}
predictions = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {"MSE": mse, "R²": r2}

# Convert results to a DataFrame
results_df = pd.DataFrame(results).T

# Step 3: Visualizations

# (a) Model Performance - MSE
plt.figure(figsize=(10, 6))
results_df['MSE'].plot(kind='bar', color='skyblue', alpha=0.8)
plt.title("Model Performance Comparison (MSE)")
plt.ylabel("Mean Squared Error")
plt.xlabel("Models")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# (b) Model Performance - R²
plt.figure(figsize=(10, 6))
results_df['R²'].plot(kind='bar', color='orange', alpha=0.8)
plt.title("Model Performance Comparison (R²)")
plt.ylabel("R² Score")
plt.xlabel("Models")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Step 4: Individual Model Analysis

# Linear Regression - Coefficients
linear_model = models["Linear Regression"]
linear_coefs = pd.Series(linear_model.coef_, index=data.drop(columns=['mpg']).columns)

plt.figure(figsize=(10, 6))
linear_coefs.plot(kind='bar', color='green', alpha=0.8)
plt.title("Linear Regression Feature Coefficients")
plt.ylabel("Coefficient Value")
plt.xlabel("Features")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Random Forest - Feature Importances
rf_model = models["Random Forest"]
rf_importances = pd.Series(rf_model.feature_importances_, index=data.drop(columns=['mpg']).columns)

plt.figure(figsize=(10, 6))
rf_importances.plot(kind='bar', color='purple', alpha=0.8)
plt.title("Random Forest Feature Importances")
plt.ylabel("Importance")
plt.xlabel("Features")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Scatter Plots for Predicted vs Actual Values

# (1) Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions["Linear Regression"], alpha=0.6, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title("Linear Regression: Predicted vs Actual")
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.grid(True)
plt.show()

# (2) Ridge Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions["Ridge Regression"], alpha=0.6, color="green")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title("Ridge Regression: Predicted vs Actual")
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.grid(True)
plt.show()

# (3) Lasso Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions["Lasso Regression"], alpha=0.6, color="orange")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title("Lasso Regression: Predicted vs Actual")
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.grid(True)
plt.show()

# (4) Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions["Random Forest"], alpha=0.6, color="purple")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title("Random Forest: Predicted vs Actual")
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.grid(True)
plt.show()

# Step 5: Print Results Summary
print("\nSummary of Model Performance:")
print(results_df)
