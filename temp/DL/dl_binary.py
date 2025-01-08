import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def convert_to_pass_fail(grades):
    return (grades >= 10).astype(int)

def create_binary_model(input_dim):
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
        
        Dense(1, activation='sigmoid')
    ])
    return model

def train_and_evaluate_binary(X, y, scenario='all'):
    # Convert grades to pass/fail
    y_binary = convert_to_pass_fail(y)
    
    # Initialize K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store results
    all_predictions = []
    all_actuals = []
    all_probas = []
    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nTraining fold {fold + 1}/5 for scenario: {scenario}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_binary[train_idx], y_binary[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Create and train model
        model = create_binary_model(X_train_scaled.shape[1])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
        ]
        
        history = model.fit(
            X_train_scaled, y_train,
            epochs=200,
            batch_size=32,
            validation_data=(X_val_scaled, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        # Get predictions
        probas = model.predict(X_val_scaled)
        predictions = (probas >= 0.5).astype(int)
        
        # Store results
        all_predictions.extend(predictions)
        all_actuals.extend(y_val)
        all_probas.extend(probas)
        
        # Calculate fold accuracy
        fold_acc = accuracy_score(y_val, predictions)
        fold_accuracies.append(fold_acc)
        print(f"Fold {fold + 1} Accuracy: {fold_acc:.4f}")
    
    return np.array(all_predictions), np.array(all_actuals), np.array(all_probas), fold_accuracies

# Load and prepare data (same as before)
data = pd.read_csv('/content/student-por.csv', sep=';')

# Process categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse_output=False, min_frequency=0.01)
encoded_features = encoder.fit_transform(data[categorical_columns])
encoded_df = pd.DataFrame(
    encoded_features, 
    columns=encoder.get_feature_names_out(categorical_columns)
)

# Prepare numeric data
numeric_data = data.drop(columns=categorical_columns)
final_data = pd.concat([numeric_data, encoded_df], axis=1)

# Run scenarios
y = final_data['G3'].values

# Scenario 1: With both G1 and G2
X_scenario1 = final_data.drop(columns=['G3'])
predictions1, actuals1, probas1, accuracies1 = train_and_evaluate_binary(X_scenario1.values, y, "with G1 and G2")

# Scenario 2: With only G1
X_scenario2 = final_data.drop(columns=['G2', 'G3'])
predictions2, actuals2, probas2, accuracies2 = train_and_evaluate_binary(X_scenario2.values, y, "with G1 only")

# Scenario 3: Without G1 and G2
X_scenario3 = final_data.drop(columns=['G1', 'G2', 'G3'])
predictions3, actuals3, probas3, accuracies3 = train_and_evaluate_binary(X_scenario3.values, y, "without G1 and G2")

# Print results for all scenarios
scenarios = ["With G1 & G2", "With G1 only", "Without G1 & G2"]
all_accuracies = [accuracies1, accuracies2, accuracies3]
all_predictions = [predictions1, predictions2, predictions3]
all_actuals = [actuals1, actuals2, actuals3]
all_probas = [probas1, probas2, probas3]

print("\nOverall Results:")
for scenario, accuracies, predictions, actuals, probas in zip(scenarios, all_accuracies, 
                                                            all_predictions, all_actuals, all_probas):
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    auc_score = roc_auc_score(actuals, probas)
    print(f"\n{scenario}:")
    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"AUC-ROC Score: {auc_score:.4f}")

# Plot ROC curves
plt.figure(figsize=(10, 6))
for scenario, actuals, probas in zip(scenarios, all_actuals, all_probas):
    fpr, tpr, _ = roc_curve(actuals, probas)
    auc = roc_auc_score(actuals, probas)
    plt.plot(fpr, tpr, label=f'{scenario} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Scenarios')
plt.legend()
plt.show()

# Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Confusion Matrices for Different Scenarios')

# Continuing from previous code...
for i, (predictions, actuals, scenario) in enumerate(zip(all_predictions, all_actuals, scenarios)):
    cm = confusion_matrix(actuals, predictions)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i])
    axes[i].set_title(scenario)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
    axes[i].set_xticklabels(['Fail', 'Pass'])
    axes[i].set_yticklabels(['Fail', 'Pass'])

plt.tight_layout()
plt.show()

    # Print detailed classification reports
for scenario, predictions, actuals in zip(scenarios, all_predictions, all_actuals):
    print(f"\nClassification Report for {scenario}:")
    print(classification_report(actuals, predictions, target_names=['Fail', 'Pass']))

    # Calculate and display class distribution
print("\nClass Distribution in Original Data:")
pass_rate = (y >= 10).mean() * 100
print(f"Pass Rate: {pass_rate:.1f}%")
print(f"Fail Rate: {100 - pass_rate:.1f}%")
