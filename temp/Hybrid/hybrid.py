import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt

class HybridRegressor:
    def __init__(self):
        self.models = {
            'nn': None,
            'xgb': XGBRegressor(n_estimators=100, learning_rate=0.1),
            'gbm': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1),
            'rf': RandomForestRegressor(n_estimators=100),
            'lasso': LassoCV(cv=5)
        }
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer()
        
    def create_nn(self, input_dim):
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
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
        return model

    def train_predict_fold(self, X_train, X_val, y_train, y_val):
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Transform target
        y_train_transformed = self.power_transformer.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        # Train Neural Network
        self.models['nn'] = self.create_nn(X_train_scaled.shape[1])
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=10, min_lr=1e-6)
        ]
        self.models['nn'].fit(
            X_train_scaled, y_train_transformed,
            epochs=200, batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # Train other models
        predictions = {}
        for name, model in self.models.items():
            if name == 'nn':
                pred = self.models['nn'].predict(X_val_scaled)
                pred = self.power_transformer.inverse_transform(pred.reshape(-1, 1)).ravel()
            else:
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_val_scaled)
            predictions[name] = pred
        
        # Ensemble predictions (weighted average)
        weights = {'nn': 0.3, 'xgb': 0.2, 'gbm': 0.2, 'rf': 0.2, 'lasso': 0.1}
        ensemble_pred = np.zeros_like(y_val)
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
            
        return predictions, ensemble_pred

def train_and_evaluate(X, y, scenario='all'):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    hybrid_model = HybridRegressor()
    
    all_predictions = {model: [] for model in hybrid_model.models.keys()}
    all_predictions['ensemble'] = []
    all_actuals = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nTraining fold {fold + 1}/5 for scenario: {scenario}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        predictions, ensemble_pred = hybrid_model.train_predict_fold(X_train, X_val, y_train, y_val)
        
        # Store predictions
        for name, pred in predictions.items():
            all_predictions[name].extend(pred)
        all_predictions['ensemble'].extend(ensemble_pred)
        all_actuals.extend(y_val)
    
    # Calculate metrics for each model
    results = {}
    for name in all_predictions.keys():
        pred = np.array(all_predictions[name])
        rmse = np.sqrt(mean_squared_error(all_actuals, pred))
        mae = mean_absolute_error(all_actuals, pred)
        r2 = r2_score(all_actuals, pred)
        results[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
    
    return results, all_predictions, all_actuals

# Load and prepare data
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

# Run scenarios
y = final_data['G3'].values

# Scenario 1: With both G1 and G2
X_scenario1 = final_data.drop(columns=['G3'])
results1, preds1, actuals1 = train_and_evaluate(X_scenario1.values, y, "with G1 and G2")

# Scenario 2: With only G1
X_scenario2 = final_data.drop(columns=['G2', 'G3'])
results2, preds2, actuals2 = train_and_evaluate(X_scenario2.values, y, "with G1 only")

# Scenario 3: Without G1 and G2
X_scenario3 = final_data.drop(columns=['G1', 'G2', 'G3'])
results3, preds3, actuals3 = train_and_evaluate(X_scenario3.values, y, "without G1 and G2")

# Print results
scenarios = ["With G1 & G2", "With G1 only", "Without G1 & G2"]
all_results = [results1, results2, results3]

print("\nOverall Results:")
for scenario, results in zip(scenarios, all_results):
    print(f"\n{scenario}:")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"R²: {metrics['r2']:.3f}")

# Plotting
plt.figure(figsize=(15, 5))
for i, (scenario, results) in enumerate(zip(scenarios, all_results)):
    plt.subplot(1, 3, i+1)
    rmse_values = [metrics['rmse'] for metrics in results.values()]
    model_names = list(results.keys())
    plt.bar(model_names, rmse_values)
    plt.title(f'RMSE by Model - {scenario}')
    plt.xticks(rotation=45)
    plt.ylabel('RMSE')
plt.tight_layout()
plt.show()
