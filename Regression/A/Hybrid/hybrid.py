# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('student-por.csv', sep=';')



# Separate features and target
target_column = 'G3'
X = data.drop(columns=[target_column])
y = data[target_column]

# Preprocess the data
# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing for numeric and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Transform the features
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ---------------------- Random Forest Model ----------------------
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_processed, y_train)
rf_predictions = rf.predict(X_test_processed)

# ---------------------- Neural Network Model ----------------------
def create_nn(input_dim):
    print(input_dim)
    model = Sequential([
        Dense(59, input_dim=input_dim, activation='sigmoid'),
        Dense(1)
    ])
    return model

# Define and compile the neural network
nn = create_nn(X_train_processed.shape[1])
nn.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', 'mse'])

# Train the neural network
nn.fit(X_train_processed, y_train, epochs=200, batch_size=16, verbose=0)

# Neural network predictions
nn_predictions = nn.predict(X_test_processed).ravel()

# ---------------------- Hybrid Model ----------------------
alpha = 0.88  # Adjust weight for Random Forest
hybrid_predictions = alpha * rf_predictions + (1 - alpha) * nn_predictions

# Evaluate the hybrid model
hybrid_rmse = np.sqrt(mean_squared_error(y_test, hybrid_predictions))
hybrid_mae = mean_absolute_error(y_test, hybrid_predictions)

print("\nHybrid Model Performance:")
print(f"Root Mean Squared Error (RMSE): {hybrid_rmse:.2f}")
print(f"Mean Absolute Error (MAE): {hybrid_mae:.2f}")
