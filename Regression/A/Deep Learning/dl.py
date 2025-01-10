import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, concatenate, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
import pandas as pd
import numpy as np

def preprocess_data(df, target_column):
    """
    Preprocess data by encoding categorical features and scaling numeric features.
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Encode categorical features
    encoders = {col: LabelEncoder() for col in categorical_cols}
    for col in categorical_cols:
        X[col] = encoders[col].fit_transform(X[col])

    # Scale numeric features
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Transform target variable
    pt = PowerTransformer()
    y = pt.fit_transform(y.values.reshape(-1, 1)).ravel()

    return X, y, encoders, scaler, pt

def build_model(input_shapes, embedding_dims):
    """
    Build a regression deep learning model that handles categorical embeddings.
    """
    inputs = []
    embeddings = []

    # Create input and embedding layers for categorical features
    for input_dim, output_dim in embedding_dims:
        input_layer = Input(shape=(1,))
        embedding_layer = Embedding(input_dim=input_dim, output_dim=output_dim)(input_layer)
        flattened_layer = Flatten()(embedding_layer)
        inputs.append(input_layer)
        embeddings.append(flattened_layer)

    # Create input for numeric features
    numeric_input = Input(shape=(input_shapes['numeric'],))
    inputs.append(numeric_input)

    # Combine embeddings and numeric features
    concatenated = concatenate(embeddings + [numeric_input])

    # Fully connected layers
    x = Dense(128, activation='relu')(concatenated)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='linear')(x)

    model = Model(inputs=inputs, outputs=output)
    return model

def train_model_with_results(df, target_column):
    """
    Preprocess the data, train the model, and calculate RMSE on the validation set.
    """
    # Preprocess data
    X, y, encoders, scaler, pt = preprocess_data(df, target_column)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['int64']).columns[:-1]  # Assuming last column is numeric
    numeric_cols = X.select_dtypes(include=['float64']).columns

    # Prepare inputs for the model
    embedding_dims = [(X[col].nunique(), min(50, (X[col].nunique() // 2) + 1)) for col in categorical_cols]
    input_shapes = {'numeric': len(numeric_cols)}

    # Build the model
    model = build_model(input_shapes, embedding_dims)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae', 'mse'])

    # Prepare inputs for training
    X_train_inputs = [X_train[col].values for col in categorical_cols] + [X_train[numeric_cols].values]
    X_val_inputs = [X_val[col].values for col in categorical_cols] + [X_val[numeric_cols].values]

    # Train the model
    history = model.fit(
        X_train_inputs, y_train,
        validation_data=(X_val_inputs, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ],
        verbose=1
    )

    # Make predictions on the validation set
    y_val_pred_transformed = model.predict(X_val_inputs).ravel()

    # Inverse transform the predictions to the original scale
    y_val_pred = pt.inverse_transform(y_val_pred_transformed.reshape(-1, 1)).ravel()
    y_val_original = pt.inverse_transform(y_val.reshape(-1, 1)).ravel()

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val_original, y_val_pred))
    print(f"\nValidation RMSE: {rmse:.2f}")

    return model, history, encoders, scaler, pt, rmse

# Example usage with a dummy dataset
if __name__ == "__main__":
    # Create a dummy dataset
    df = pd.read_csv('student-por.csv', sep=';')


    # Train the model
    model, history, encoders, scaler, pt, rmse = train_model_with_results(df, target_column='G3')

    print("\nTraining Completed!")
    final_val_loss = history.history['val_loss'][-1]
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Epochs Completed: {len(history.history['loss'])}")

    # Print the RMSE
    print(f"\nValidation RMSE: {rmse:.2f}")
