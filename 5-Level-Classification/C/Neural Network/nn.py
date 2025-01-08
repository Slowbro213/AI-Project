
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, recall_score, precision_score, recall_score, f1_score, accuracy_score
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('https://support.curepharm.org/student-por.csv', sep=';')

# Define class ranges and map grades to class labels
def map_grades_to_classes(grade):
    if 0 <= grade <= 4:
        return 0  # "Level 1: Very poor"
    elif 5 <= grade <= 9:
        return 1  # "Level 2: Poor"
    elif 10 <= grade <= 12:
        return 2  # "Level 3: Average"
    elif 13 <= grade <= 16:
        return 3  # "Level 4: Good"
    elif 17 <= grade <= 20:
        return 4  # "Level 5: Excellent"

# Apply the mapping to create class labels
data['G3_class'] = data['G3'].apply(map_grades_to_classes)

# Preprocess the data
# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# OneHotEncode categorical columns
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(data[categorical_columns])

# Create a new DataFrame for encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Drop original categorical columns and concatenate encoded columns
data = data.drop(columns=categorical_columns).reset_index(drop=True)
data = pd.concat([data, encoded_df], axis=1)

# Define features (X) and target (y)
X = data.drop(columns=["G1", "G2", "G3", "G3_class"])
y = data['G3_class']

# One-hot encode the target variable for classification
y_one_hot = tf.keras.utils.to_categorical(y, num_classes=5)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('Number of features after one_hot_encoding:', X_train.shape[1])

# Build the neural network model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(5, activation='softmax')  # Output layer for 5 classes
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',  # Loss function for multi-class classification
              metrics=['accuracy'])  # Accuracy as evaluation metric

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Predict on test data
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # Predicted class labels
y_test_actual = np.argmax(y_test, axis=1)  # True class labels

# Classification metrics
precision = precision_score(y_test_actual, y_pred, average='weighted')
recall = recall_score(y_test_actual, y_pred, average='weighted')
f1 = f1_score(y_test_actual, y_pred, average='weighted')
accuracy = accuracy_score(y_test_actual, y_pred)

print("\nClassification Metrics:")
print(f"Precision (Weighted): {precision:.2f}")
print(f"Recall (Weighted): {recall:.2f}")
print(f"F1-Score (Weighted): {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")

# Pearson Correlation Coefficient (PCC)
# For PCC, convert classes to their numeric range representations
class_to_grade_map = {0: 2, 1: 7, 2: 11, 3: 14.5, 4: 18.5}  # Approximate class midpoints
y_test_actual_grades = [class_to_grade_map[label] for label in y_test_actual]
y_pred_grades = [class_to_grade_map[label] for label in y_pred]

pcc, _ = pearsonr(y_test_actual_grades, y_pred_grades)
print(f"Pearson Correlation Coefficient (PCC): {pcc:.2f}")

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test_actual, y_pred, display_labels=[
    "Very poor", "Poor", "Average", "Good", "Excellent"
])
plt.title('Confusion Matrix')
plt.show()

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()
plt.show()
