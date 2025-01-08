import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score
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

# Step 2: Define features and target (G3 for regression)
X = df.drop(columns=['G1','G2','G3'])  
y = df['G3'] 

# Step 3: Scale the features using Z-score normalization (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA to reduce dimensionality
pca = PCA(n_components=23)  
X_pca = pca.fit_transform(X_scaled)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

print(f"Training data size: {X_train.shape}, Test data size: {X_test.shape}")

# Step 6: Train the Gaussian Naive Bayes model for regression
gnb = GaussianNB()
gnb.fit(X_train, y_train) 

# Step 7: Make predictions on the test set
y_pred = gnb.predict(X_test)  

# Step 8: Evaluate the model using regression metrics
mse = mean_squared_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)  

# Print metrics
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Step 9: Visualize the predicted vs actual values
plt.scatter(y_test, y_pred)
plt.plot([0, 20], [0, 20], 'k--', lw=2) 
plt.xlabel('True Values (G3)')
plt.ylabel('Predicted Values (G3)')
plt.title('True vs Predicted Values')
plt.show()
