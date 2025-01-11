import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def convert_to_grade_levels(grades):
    levels = np.zeros_like(grades)
    levels[(grades >= 0) & (grades <= 9)] = 1  # Very poor
    levels[(grades > 9) & (grades <= 11)] = 2   # Poor
    levels[(grades > 11) & (grades <= 13)] = 3  # Average
    levels[(grades > 13) & (grades <= 15)] = 4 # Good
    levels[(grades > 15) & (grades <= 20)] = 5 # Excellent
    return levels

def create_classification_model(input_dim, num_classes):
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
        
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_and_evaluate_scenario(X, y, scenario='all'):
    # Convert grades to class levels
    y_classes = convert_to_grade_levels(y)
    
    # Initialize K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results
    all_predictions = []
    all_actuals = []
    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nTraining fold {fold + 1}/5 for scenario: {scenario}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_classes[train_idx], y_classes[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Convert to categorical
        y_train_cat = tf.keras.utils.to_categorical(y_train - 1, num_classes=5)
        y_val_cat = tf.keras.utils.to_categorical(y_val - 1, num_classes=5)
        
        # Create and train model
        model = create_classification_model(X_train_scaled.shape[1], 5)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
        ]
        
        history = model.fit(
            X_train_scaled, y_train_cat,
            epochs=200,
            batch_size=32,
            validation_data=(X_val_scaled, y_val_cat),
            callbacks=callbacks,
            verbose=0
        )
        
        # Get predictions
        pred_probs = model.predict(X_val_scaled)
        predictions = np.argmax(pred_probs, axis=1) + 1
        
        # Store results
        all_predictions.extend(predictions)
        all_actuals.extend(y_val)
        
        # Calculate fold accuracy
        fold_acc = accuracy_score(y_val, predictions)
        fold_accuracies.append(fold_acc)
        print(f"Fold {fold + 1} Accuracy: {fold_acc:.4f}")
    
    return np.array(all_predictions), np.array(all_actuals), fold_accuracies

# Load data
data = pd.read_csv('/content/student-por.csv', sep=';')

# Process categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, min_frequency=0.01)
encoded_features = encoder.fit_transform(data[categorical_columns])
encoded_df = pd.DataFrame(
    encoded_features, 
    columns=encoder.get_feature_names_out(categorical_columns)
)

# Prepare numeric data
numeric_data = data.drop(columns=categorical_columns)
final_data = pd.concat([numeric_data, encoded_df], axis=1)

# Scenario 1: With both G1 and G2
X_scenario1 = final_data.drop(columns=['G3'])
y = final_data['G3'].values
predictions1, actuals1, accuracies1 = train_and_evaluate_scenario(X_scenario1.values, y, "with G1 and G2")

# Scenario 2: With only G1

# Print results for all scenarios
scenarios = ["With G1 & G2", "With G1 only", "Without G1 & G2"]
all_accuracies = [accuracies1, accuracies2, accuracies3]

print("\nOverall Results:")
for scenario, accuracies in zip(scenarios, all_accuracies):
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"\n{scenario}:")
    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
# Plot confusion matrices for all scenarios
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Confusion Matrices for Different Scenarios')

for i, (predictions, actuals, scenario) in enumerate(zip(
    [predictions1, predictions2, predictions3],
    [actuals1, actuals2, actuals3],
    scenarios
)):
    cm = confusion_matrix(actuals, predictions)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i])
    axes[i].set_title(scenario)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Print detailed classification reports
for i, (predictions, actuals, scenario) in enumerate(zip(
    [predictions1, predictions2, predictions3],
    [actuals1, actuals2, actuals3],
    scenarios
)):
    print(f"\nClassification Report for {scenario}:")
    print(classification_report(actuals, predictions, 
                              target_names=['Very Poor', 'Poor', 'Average', 'Good', 'Excellent']))
