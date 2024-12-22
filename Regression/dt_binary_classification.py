import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import export_graphviz

# Load datasets
data_mat = pd.read_csv("student-mat.csv", sep=";")
data_por = pd.read_csv("student-por.csv", sep=";")

# Merge datasets based on common attributes
common_columns = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]
data = pd.merge(data_mat, data_por, on=common_columns, suffixes=("_mat", "_por"))

# Encode categorical variables as numeric
data = pd.get_dummies(data, drop_first=True)

# Convert the target variable (G3_mat) into a binary classification problem
data['G3_binary'] = (data['G3_mat'] >= 10).astype(int)  # Pass/Fail classification (1 for pass, 0 for fail)

# Choose target (G3_binary) and relevant features
features = ["studytime_mat", "failures_mat", "famsup_mat_yes", "traveltime_mat", "absences_mat", "G1_mat", "G2_mat"]
X = data[features]
y = data["G3_binary"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
classifier = DecisionTreeClassifier(random_state=42, max_depth=3)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize the decision tree with Graphviz
feature_names = features

# Export the decision tree in DOT format
dot_data = export_graphviz(classifier, out_file=None, feature_names=feature_names, filled=True, rounded=True, special_characters=True)

# Use Graphviz to render the decision tree
graph = graphviz.Source(dot_data)
graph.render("decision_tree_binary", format="png", cleanup=True)
print("Binary classification decision tree visualization saved as 'decision_tree_binary.png'.")
