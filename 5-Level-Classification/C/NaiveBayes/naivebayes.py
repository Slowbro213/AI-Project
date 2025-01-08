import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns

# Read the dataset
df = pd.read_csv("student-por.csv", sep=";")

# Define categorical columns for encoding
binary_ordinal_columns = ['sex', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities',
                          'nursery', 'higher', 'internet', 'romantic']

# Apply Label Encoding to binary/ordinal columns
label_encoder = LabelEncoder()

for col in binary_ordinal_columns:
    df[col] = label_encoder.fit_transform(df[col])

# For nominal columns, apply One-Hot Encoding
nominal_columns = ['school', 'address', 'Mjob', 'Fjob', 'reason', 'guardian']

df = pd.get_dummies(df, columns=nominal_columns, drop_first=True)

# Step 2: Define features (G1, G2) and target (5-level grade classification)
X = df.drop(columns=['G1','G2','G3']) 

# Convert 'G3' into a 5-level classification based on the Portuguese grading system
bins = [0, 10, 14, 16, 18, 20]  # Grade boundaries for the Portuguese grading system
labels = ['Fail', 'Sufficient', 'Satisfactory', 'Good', 'Excellent']  # Class labels
df['G3_category'] = pd.cut(df['G3'], bins=bins, labels=labels, right=True)

# Target variable for 5-level classification
y = df['G3_category']

# Step 3: Scale the features using Z-score normalization (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA to reduce dimensionality
pca = PCA(n_components=23)  # Keep 23 principal components (you can adjust this number)
X_pca = pca.fit_transform(X_scaled)

# Check the shape of X_pca to ensure it's 2D
print(f"Shape of X_pca: {X_pca.shape}")  # It should print (n_samples, n_components)

# Ensure the target variable is encoded into numeric values
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)  # Encode target labels

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.3, random_state=42)

# Check the shapes of X_train, X_test, y_train, and y_test to make sure they are correct
print(f"Shape of X_train: {X_train.shape}, Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}, Shape of y_test: {y_test.shape}")

# Step 6: Train the Gaussian Naive Bayes model using the PCA-transformed features
gnb = GaussianNB()
gnb.fit(X_train, y_train)  # Fit the model on the PCA-transformed training data

# Step 7: Make predictions on the test set
y_pred = gnb.predict(X_test)  # Predict on PCA-transformed test data (X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision (weighted): {precision:.2f}")
print(f"Recall (weighted): {recall:.2f}")
print(f"F1 Score (weighted): {f1:.2f}\n")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Step 9: Plot the confusion matrix using a heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder_y.classes_, yticklabels=label_encoder_y.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
