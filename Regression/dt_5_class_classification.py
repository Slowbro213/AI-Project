import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import graphviz
from sklearn.tree import export_graphviz

# Load datasets
data_mat = pd.read_csv('student-mat.csv', sep=';')
data_por = pd.read_csv('student-por.csv', sep=';')

# Merge datasets
common_columns = ["school", "sex", "age", "address", "famsize", "Pstatus",
                  "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]
data = pd.merge(data_mat, data_por, on=common_columns, suffixes=("_mat", "_por"))

# Create 5-class target variable based on the average G3
data['G3_avg'] = data[['G3_mat', 'G3_por']].mean(axis=1)
bins = [0, 9, 11, 13, 15, 20]
labels = ['Fail', 'Sufficient', 'Satisfactory', 'Good', 'Excellent']
data['G3_class'] = pd.cut(data['G3_avg'], bins=bins, labels=labels, right=False)

# Define features and target
X = data.drop(columns=['G3_mat', 'G3_por', 'G3_avg', 'G3_class'])
y = data['G3_class']

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=5)  # Adjust max_depth as needed
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Visualize the Decision Tree (Graphviz for detailed view)
feature_names = X.columns
class_names = labels

dot_data = export_graphviz(
    dt_classifier,
    out_file=None,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_5_class", format="png", cleanup=True)
print("Decision Tree visualization saved as 'decision_tree_5_class.png'.")
