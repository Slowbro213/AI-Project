#ZSCORE WITH G1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score,recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns

# Read the dataset
df = pd.read_csv("student-por.csv", sep=";")

# Define categorical columns for encoding
binary_ordinal_columns = ['sex', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities',
                          'nursery', 'higher', 'internet', 'romantic']

label_encoder = LabelEncoder()

for col in binary_ordinal_columns:
    df[col] = label_encoder.fit_transform(df[col])

nominal_columns = ['school', 'address', 'Mjob', 'Fjob', 'reason', 'guardian']

df = pd.get_dummies(df, columns=nominal_columns, drop_first=True)


# Step 2: Define features (G1, G2) and target (pass_fail)
X = df.drop(columns=['G2','G3'])  
y = (df['G3'] >= 10).astype(int)  

# Step 3: Scale the features using Z-score normalization (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA to reduce dimensionality
pca = PCA(n_components=23)  
X_pca = pca.fit_transform(X_scaled)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

print(f"Training data size: {X_train.shape}, Test data size: {X_test.shape}")

# Step 6: Train the Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = gnb.predict(X_test)

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
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
