# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('student-por.csv', sep=';')

# Preprocess the data
# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# OneHotEncode categorical columns
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(data[categorical_columns])

# Create a new DataFrame for encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Drop original categorical columns and concatenate encoded columns
data = data.drop(columns=categorical_columns).reset_index(drop=True)
data = pd.concat([data, encoded_df], axis=1)

# Remove 'G1' and 'G2' from features
if 'G1' in data.columns:
    data = data.drop(columns=['G1'])
if 'G2' in data.columns:
    data = data.drop(columns=['G2'])

# Define five-level classification target
def categorize_grade(grade):
    if grade < 5:
        return 0  # Very Poor
    elif grade < 10:
        return 1  # Poor
    elif grade < 15:
        return 2  # Average
    elif grade < 18:
        return 3  # Good
    else:
        return 4  # Excellent

data['Grade_Level'] = data['G3'].apply(categorize_grade)

# Define features (X) and target (y)
X = data.drop(columns=['G3', 'Grade_Level'])
y = data['Grade_Level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest Classifier
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy:.2f}\n")

# Detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Very Poor", "Poor", "Average", "Good", "Excellent"]))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot a beautiful confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Very Poor", "Poor", "Average", "Good", "Excellent"],
    yticklabels=["Very Poor", "Poor", "Average", "Good", "Excellent"]
)
plt.title("Confusion Matrix for Five-Level Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
