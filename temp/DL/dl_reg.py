import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PowerTransformer

def load_and_preprocess_data(filepath, target_column='G3',exclude=''):
    # Load the data
    data = pd.read_csv(filepath, sep=';')
    
    # Create interaction features for numeric columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    numeric_data = data[numeric_cols]
    
    # Create polynomial features for numeric columns
    poly = PolynomialFeatures(degree=2, include_bias=False)
    numeric_poly = poly.fit_transform(numeric_data)
    numeric_poly_df = pd.DataFrame(
        numeric_poly[:, len(numeric_cols):],  # Skip original features
        columns=[f'poly_{i}' for i in range(numeric_poly.shape[1] - len(numeric_cols))]
    )
    
    # Preprocess categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    # OneHotEncode categorical columns with handling for rare categories
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, min_frequency=0.01)  # Remove rare categories
    encoded_features = encoder.fit_transform(data[categorical_columns])
    
    # Create DataFrame for encoded features
    encoded_df = pd.DataFrame(
        encoded_features, 
        columns=encoder.get_feature_names_out(categorical_columns)
    )
    
    # Combine all features
    data = pd.concat([
        data[numeric_cols],
        numeric_poly_df,
        encoded_df
    ], axis=1)
    
    # Define features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Apply power transform to target variable
    pt = PowerTransformer()
    y_transformed = pt.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    return X, y_transformed, pt

def create_deep_model(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(1)
    ])
    return model

# Load and preprocess data
X, y_transformed, power_transformer = load_and_preprocess_data('/content/student-por.csv')

# Feature selection
selector = SelectKBest(score_func=mutual_info_regression, k=25)
X_selected = selector.fit_transform(X, y_transformed)
selected_features = X.columns[selector.get_support()].tolist()

# Initialize K-Fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Lists to store predictions and actual values
all_predictions = []
all_actuals = []

# Train models with cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected)):
    print(f"\nTraining fold {fold + 1}/{n_splits}")
    
    # Split data
    X_train, X_val = X_selected[train_idx], X_selected[val_idx]
    y_train, y_val = y_transformed[train_idx], y_transformed[val_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create and train neural network
    model = create_deep_model(X_train_scaled.shape[1])
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='huber',
        metrics=['mae', 'mse']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
    ]
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=300,
        batch_size=32,
        validation_data=(X_val_scaled, y_val),
        callbacks=callbacks,
        verbose=0
    )
    
    # Get predictions
    fold_predictions = model.predict(X_val_scaled)
    
    # Store predictions and actual values
    all_predictions.extend(fold_predictions.ravel())
    all_actuals.extend(y_val)

# Inverse transform predictions and actuals
predictions_original = power_transformer.inverse_transform(np.array(all_predictions).reshape(-1, 1)).ravel()
actuals_original = power_transformer.inverse_transform(np.array(all_actuals).reshape(-1, 1)).ravel()

# Calculate metrics
mse = mean_squared_error(actuals_original, predictions_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actuals_original, predictions_original)
r2 = r2_score(actuals_original, predictions_original)

print("\nFinal Model Performance Metrics:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.3f}")

# Plot predictions vs actuals
plt.figure(figsize=(10, 6))
plt.scatter(actuals_original, predictions_original, alpha=0.5)
plt.plot([0, 20], [0, 20], 'r--')
plt.xlabel('Actual Grades')
plt.ylabel('Predicted Grades')
plt.title('Actual vs Predicted Grades')
plt.tight_layout()
plt.show()

# Plot prediction error distribution
errors = predictions_original - actuals_original
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, edgecolor='black')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.tight_layout()
plt.show()
