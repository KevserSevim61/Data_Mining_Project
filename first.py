# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# 1. Importing and Initial Exploration of the Dataset
# Load the dataset
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Display the first few rows of the dataset
print("First 5 Rows of the Dataset:")
print(df.head())

# Check the general information of the dataset
print("\nDataset Information:")
print(df.info())

# Get a statistical summary of the dataset
print("\nStatistical Summary of the Dataset:")
print(df.describe())

# 2. Checking and Handling Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Visualizing the Distribution of the Target Variable
sns.countplot(x='DEATH_EVENT', data=df)
plt.title("Distribution of the Target Variable (DEATH_EVENT)")
plt.show()

# 4. Visualizing the Correlation Matrix
# Compute the correlation matrix
correlation_matrix = df.corr()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar=True)
plt.title("Feature Correlation")
plt.show()

# 5. Data Preprocessing
# Separate the target variable and features
X = df.drop('DEATH_EVENT', axis=1)  # Features
y = df['DEATH_EVENT']  # Target variable

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Splitting the Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# 7. Training and Evaluating with Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_dt = dt_model.predict(X_test)
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]

# Performance metrics
print("\nDecision Tree Model Performance:")
print("Accuracy Score:", accuracy_score(y_test, y_pred_dt))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba_dt))
print(classification_report(y_test, y_pred_dt))

# Confusion matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues')
plt.title("Decision Tree - Confusion Matrix")
plt.show()

# 9. Training and Evaluating with Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Performance metrics
print("\nRandom Forest Model Performance:")
print("Accuracy Score:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba_rf))
print(classification_report(y_test, y_pred_rf))

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest - Confusion Matrix")
plt.show()

# 10. Comparing Model Performances
results = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_rf)],
    "ROC-AUC": [roc_auc_score(y_test, y_pred_proba_dt), roc_auc_score(y_test, y_pred_proba_rf)]
})

print("\nModel Comparison Results:")
print(results)
