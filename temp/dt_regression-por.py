import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import export_graphviz

# Load the Portuguese dataset
data_por = pd.read_csv("student-por.csv", sep=";")

# Encode categorical variables as numeric
data = pd.get_dummies(data_por, drop_first=True)

# Choose target (G3) and relevant features
features = ["studytime", "failures", "famsup_yes", "traveltime", "absences", "G1", "G2"]
X = data[features]
y = data["G3"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree regressor
regressor = DecisionTreeRegressor(random_state=42, max_depth=6)
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

r2 = r2_score(y_test, y_pred)
print(f"R² Score (Accuracy Equivalent): {r2:.2f}")

# Visualize the decision tree with Graphviz
feature_names = features

# Export the decision tree in DOT format
dot_data = export_graphviz(regressor, out_file=None, feature_names=feature_names, filled=True, rounded=True, special_characters=True)

# Use Graphviz to render the decision tree
graph = graphviz.Source(dot_data)
graph.render("decision_tree_por", format="png", cleanup=True)
print("Decision tree visualization saved as 'decision_tree_por.png'.")
