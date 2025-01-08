# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC  # Support Vector Classification for classification task
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('student-por.csv', sep=';')

# Create a 5-level classification target based on G3
bins = [0, 9, 11, 13, 15, 20]  # Define the bin edges for the 5 classes
labels = ['Fail', 'Sufficient', 'Satisfactory', 'Good', 'Excellent']  # The corresponding labels
data['G3_class'] = pd.cut(data['G3'], bins=bins, labels=labels, right=False)

# Drop the original G3 column and create new features for classification
target_column = 'G3_class'
X = data.drop(columns=[target_column, 'G3'])  # Drop the target and the original G3 column
y = data[target_column]

# Preprocess the categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(X[categorical_columns])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Drop original categorical columns and concatenate the encoded ones
X = X.drop(columns=categorical_columns).reset_index(drop=True)
X = pd.concat([X, encoded_df], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Grid Search to find the best hyperparameters for the SVM model
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1, 10],
    'kernel': [ 'rbf', 'poly']  # Trying different kernels
}

svm_model = SVC()
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Train the best SVM model
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")
best_svm_model = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
best_svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = best_svm_model.predict(X_test)

# Print the classification report and accuracy
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Plot confusion matrix (optional)
from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
