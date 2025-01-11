import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('student-por.csv', sep=';')

# Drop G1 and G2 grades
data = data.drop(columns=['G1', 'G2'])

# Binary classification: Pass (>=10) or Fail (<10)
data['G3_binary'] = (data['G3'] >= 10).astype(int)

# One-hot encode categorical variables using pd.get_dummies
data = pd.get_dummies(data, drop_first=True)

# Define features and target
target_column = 'G3_binary'
X = data.drop(columns=['G3', 'G3_binary'])
y = data['G3_binary']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Decision Tree Classifier ----------------
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=2)
dt_classifier.fit(X_train, y_train)
dt_preds = dt_classifier.predict_proba(X_test)[:, 1]  # Probabilities for class 1

# ---------------- Neural Network ----------------
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network
nn_model = Sequential([
        Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
nn_model.fit(X_train_scaled, y_train, epochs=500, batch_size=32, verbose=0)

# Neural network predictions
nn_preds = nn_model.predict(X_test_scaled).ravel()  # Probabilities for class 1

# ---------------- Hybrid Model ----------------
# Weighted average of predictions
hybrid_preds_probs = np.where(dt_preds > nn_preds, dt_preds, nn_preds)
hybrid_preds = (hybrid_preds_probs >= 0.5).astype(int)

# ---------------- Evaluation ----------------
def evaluate_model(y_true, y_pred, name="Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n{name} Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

evaluate_model(y_test, (dt_preds >= 0.5).astype(int), "Decision Tree")
evaluate_model(y_test, (nn_preds >= 0.5).astype(int), "Neural Network")
evaluate_model(y_test, hybrid_preds, "Hybrid Model")

# Confusion Matrix for Hybrid Model
conf_matrix = confusion_matrix(y_test, hybrid_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.title("Confusion Matrix for Hybrid Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
