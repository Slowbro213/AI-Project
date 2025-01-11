import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('student-por.csv', sep=';')

# First create the class labels
bins = [0, 8, 11, 14, 16, 20]
labels = ['Fail', 'Sufficient', 'Satisfactory', 'Good', 'Excellent']
data['G3_class'] = pd.cut(data['G3'], bins=bins, labels=labels, right=False)

# Then create features (before one-hot encoding)
feature_cols = [col for col in data.columns if col not in ['G2','G3', 'G3_class']]
X = data[feature_cols]
y = data['G3_class']

# One-hot encode the feature columns
X = pd.get_dummies(X, drop_first=True)

# Convert labels to numeric
le = LabelEncoder()
y_numeric = le.fit_transform(y)
num_classes = len(labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric)

# Convert to one-hot encoded format for neural network
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# ---------------- Decision Tree Classifier ----------------
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=4)
dt_classifier.fit(X_train, y_train)
dt_probs = dt_classifier.predict_proba(X_test)
dt_preds = dt_classifier.predict(X_test)

# ---------------- Neural Network ----------------
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network for multi-class classification
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
    
    Dense(num_classes, activation='softmax')
])

nn_model.compile(optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

nn_model.fit(X_train_scaled, y_train_cat, 
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1)

# Neural network predictions
nn_probs = nn_model.predict(X_test_scaled)
nn_preds = np.argmax(nn_probs, axis=1)

# ---------------- Hybrid Model ----------------
def combine_predictions(dt_probs, nn_probs, alpha=0.5):
    # Weighted average of probabilities
    hybrid_probs = alpha * dt_probs + (1 - alpha) * nn_probs
    
    # For each sample, if models strongly agree (both have high confidence),
    # increase the weight of their consensus
    model_agreement = np.sum(dt_probs * nn_probs, axis=1)
    boost_mask = model_agreement > 0.3
    
    # Boost the probabilities where models agree
    hybrid_probs[boost_mask] *= 1.2
    # Normalize probabilities
    hybrid_probs = hybrid_probs / np.sum(hybrid_probs, axis=1, keepdims=True)
    
    return hybrid_probs

# Get hybrid predictions
hybrid_probs = combine_predictions(dt_probs, nn_probs, alpha=0.6)
hybrid_preds = np.argmax(hybrid_probs, axis=1)

# ---------------- Evaluation ----------------
def evaluate_model(y_true, y_pred, name="Model"):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"\n{name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return accuracy

# Evaluate all models
dt_accuracy = evaluate_model(y_test, dt_preds, "Decision Tree")
nn_accuracy = evaluate_model(y_test, nn_preds, "Neural Network")
hybrid_accuracy = evaluate_model(y_test, hybrid_preds, "Hybrid Model")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot confusion matrices
for ax, (preds, title, cmap) in zip(
    axes,
    [(dt_preds, 'Decision Tree', 'Blues'),
     (nn_preds, 'Neural Network', 'Greens'),
     (hybrid_preds, 'Hybrid Model', 'Oranges')]
):
    sns.heatmap(
        confusion_matrix(y_test, preds),
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_title(f'{title}\nAccuracy: {accuracy_score(y_test, preds):.4f}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.show()
