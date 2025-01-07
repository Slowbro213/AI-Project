# Import necessary libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('student-por.csv', sep=';')

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
target_column = 'G3'  # Final grade
X = data.drop(columns=[target_column])
y = data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest Regressor with the best parameters
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Root Mean Squared Error (MSE): {(mse**0.5):.2f}, RMSE of the paper: 1.32")
print(f"Mean Absolute Error (MAE): {mae:.2f}")