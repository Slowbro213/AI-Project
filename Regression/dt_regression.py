import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error
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

# Choose target (G3) and relevant features
features = ["studytime_mat", "failures_mat", "famsup_mat_yes", "traveltime_mat", "absences_mat", "G1_mat", "G2_mat"]
X = data[features]
y = data["G3_mat"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree regressor
regressor = DecisionTreeRegressor(random_state=42, max_depth=3)
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualize the decision tree with Graphviz
feature_names = features

# Export the decision tree in DOT format
dot_data = export_graphviz(regressor, out_file=None, feature_names=feature_names, filled=True, rounded=True, special_characters=True)

# Use Graphviz to render the decision tree
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png", cleanup=True)
print("Decision tree visualization saved as 'decision_tree.png'.")