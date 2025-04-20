import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
data = pd.read_csv("nutrition.csv", sep="\t", nrows=10000, low_memory=False)

# Selecting features and target
data_features = data[["energy_100g", "saturated-fat_100g", "carbohydrates_100g", "fiber_100g", "fat_100g", "proteins_100g", "salt_100g"]]
target = data['nutrition-score-fr_100g']

# Drop NaN values in target
data = data.dropna(subset=['nutrition-score-uk_100g', 'nutrition-score-fr_100g'])
data_features = data_features.fillna(data_features.mean())
target = target.fillna(target.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(data_features, target, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=500, random_state=42),
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1),
    "SVM": SVR(kernel='rbf')
}

# Training and Evaluation
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        "R2 Score": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred)
    }

# Print results
for name, metrics in results.items():
    print(f"{name}: R2 Score = {metrics['R2 Score']:.4f}, MAE = {metrics['MAE']:.4f}, MSE = {metrics['MSE']:.4f}")

# Custom Prediction
custom_input = pd.DataFrame({
    "energy_100g": [23300],
    "saturated-fat_100g": [0],
    "carbohydrates_100g": [0],
    "fiber_100g": [0],
    "fat_100g": [0],
    "proteins_100g": [100],
    "salt_100g": [0]
})

custom_input_scaled = scaler.transform(custom_input)
custom_prediction = models["Random Forest"].predict(custom_input_scaled)
print(f"Custom Input Prediction: {custom_prediction[0]}")
