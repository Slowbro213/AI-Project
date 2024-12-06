# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam , SGD

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/content/student-por.csv', sep=';')

# Preprocess the data
# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# OneHotEncode categorical columns
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(data[categorical_columns])

# Create a new DataFrame for encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Drop original categorical columns and concatenate encoded columns
data = data.drop(columns=categorical_columns).reset_index(drop=True)
data = pd.concat([data, encoded_df], axis=1)

# Define features (X) and target (y)
target_column = 'G3'  # Assuming 'G3' is the target column for final grade
X = data.drop(columns=[target_column])
y = data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('number of features after one_hot_encoding:',X_train.shape[1])
# Build the neural network model
model = Sequential([
    Dense(59, input_dim=X_train.shape[1], activation='sigmoid'),
    Dense(1)  # Single output for regression
])
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mse',  # Mean Squared Error for regression
              metrics=['mse'])  # Mean Absolute Error as a metric

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {mae:.2f}")

# Predict on test data
y_pred = model.predict(X_test)

# Calculate additional metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Root Mean Squared Error (RMSE): {(mse**0.5):.2f}, RMSE of the paper: 1.36")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['mse'], label='Training MAE')
plt.plot(history.history['val_mse'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Training and Validation MAE over Epochs')
plt.legend()
plt.show()

# Plot predictions vs actual values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Grades (G3)')
plt.ylabel('Predicted Grades (G3)')
plt.title('Actual vs Predicted Grades')
plt.show()
