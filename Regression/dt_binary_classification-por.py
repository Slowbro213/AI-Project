import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import export_graphviz

# Load the Portuguese dataset
data = pd.read_csv("student-por.csv", sep=";")

# Convert the target variable (G3) into a binary classification problem
# Passing if G3 >= 10, failing otherwise
data['G3_binary'] = (data['G3'] >= 10).astype(int)

# Choose features and target
# Use features present in the dataset
features = ["studytime", "failures", "famsup", "traveltime", "absences", "G1", "G2"]
X = data[features]
y = data["G3_binary"]

# Encode categorical variables as numeric
X = pd.get_dummies(X, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
classifier = DecisionTreeClassifier(random_state=42, max_depth=4) 
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize the decision tree with Graphviz
feature_names = X.columns
dot_data = export_graphviz(
    classifier,
    out_file=None,
    feature_names=feature_names,
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_binary", format="png", cleanup=True)
print("Binary classification decision tree visualization saved as 'decision_tree_binary.png'.")