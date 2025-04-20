import OcrToTableTool as ottt
import TableExtractor as te
import TableLinesRemover as tlr
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




path_to_image = "./OCR/image/img4.jpg"
table_extractor = te.TableExtractor(path_to_image)
perspective_corrected_image = table_extractor.execute()
#cv2.imshow("perspective_corrected_image", perspective_corrected_image)

lines_remover = tlr.TableLinesRemover(perspective_corrected_image)
image_without_lines = lines_remover.execute()
#cv2.imshow("image_without_lines", image_without_lines)

ocr_tool = ottt.OcrToTableTool(image_without_lines, perspective_corrected_image)
table_data = ocr_tool.execute()
# Print structured table data
print("Formatted Table Data:", table_data)

#cv2.destroyAllWindows()


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


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_ocr_data_for_model(ocr_table_data, scaler):
    """
    Transform OCR table data into a format suitable for the ML model prediction.
    
    Args:
        ocr_table_data (list): The OCR output in list format
        scaler (StandardScaler): The fitted scaler used for the training data
        
    Returns:
        prediction_ready_data (DataFrame): Data ready for model prediction
    """
    # Initialize a dictionary to store nutritional values
    nutrition_dict = {
        "energy_100g": 0,
        "saturated-fat_100g": 0,
        "carbohydrates_100g": 0,
        "fiber_100g": 0,
        "fat_100g": 0,
        "proteins_100g": 0,
        "salt_100g": 0
    }
    
    # Map OCR labels to model feature names
    label_mapping = {
        "ENERGY": "energy_100g",
        "CARBOHYDRATE": "carbohydrates_100g",
        "DIETARY FIBRE": "fiber_100g",
        "FAT": "fat_100g",
        ". PROTEIN": "proteins_100g",  # Fixed: OCR label -> feature name
        "SALT": "salt_100g",
        "SATURATED FATTY ACIDS": "saturated-fat_100g"
    }
    
    # Extract values from OCR data
    for row in ocr_table_data:
        if len(row) == 2:  # Only process rows with label-value pairs
            label = row[0].strip()
            value_str = row[1].strip()
            
            # Try to extract numerical value
            try:
                # Extract number from the value (remove units like 'g' or 'Kcal')
                numeric_value = ''.join(c for c in value_str if c.isdigit() or c == '.' or c == '-')
                value = float(numeric_value)
                
                # Map to the appropriate feature
                for ocr_label, feature_name in label_mapping.items():
                    if ocr_label in label:
                        nutrition_dict[feature_name] = value
                        break
            except ValueError:
                print(f"Could not convert value for {label}: {value_str}")
    
    # Create DataFrame
    df = pd.DataFrame([nutrition_dict])
    
    # Scale the data
    scaled_data = scaler.transform(df)
    
    return df, scaled_data

def predict_nutrition_score(ocr_data, model, scaler):
    """
    Make a prediction using OCR data
    
    Args:
        ocr_data (list): OCR output data
        model: Trained ML model
        scaler: Fitted scaler
        
    Returns:
        prediction: The predicted nutrition score
    """
    # Prepare OCR data for prediction
    original_df, scaled_data = prepare_ocr_data_for_model(ocr_data, scaler)
    
    # Make prediction
    prediction = model.predict(scaled_data)
    
    return {
        "input_features": original_df.to_dict(orient='records')[0],
        "predicted_score": float(prediction[0])
    }

# Example usage
def process_ocr_output(ocr_table_data, model, scaler):
    
    """
    Main function to process OCR output and predict nutrition score
    
    Args:
        ocr_table_data (list): OCR output as a list of lists
        model: Trained ML model
        scaler: Fitted scaler
        
    Returns:
        dict: Prediction results with input features and predicted score
    """
    print("Processing OCR data...")
    print(f"Input OCR data: {ocr_table_data}")
    
    result = predict_nutrition_score(ocr_table_data, model, scaler)
    
    print("\nPrediction Results:")
    print(f"Input Features:")
    for feature, value in result["input_features"].items():
        print(f"  {feature}: {value}")
    print(f"Predicted Nutrition Score: {result['predicted_score']:.2f}")
    
    return result

# After your OCR code and model training

# Get your best model (Random Forest in this case)
best_model = models["Random Forest"]

# Your OCR result
table_data = [['NUTRITIONAL INFORMATION'], 
              ['PER 100g PRODUCT'], 
              ['ENERGY', '457 Kcal'], 
              ['CARBOHYDRATE', '60.2 g'], 
              ['TOTAL SUGAR', '1.99'], 
              ['FAT', '17.6 g'], 
              ['DIETARY FIBRE', '9.79'], 
              ['. PROTEIN', '9.19'], 
              ['SALT', '8.3 g'], 
              ['SATURATED FATTY ACIDS', '619.']]

# Make prediction
result = process_ocr_output(table_data, best_model, scaler)

print("RESLUT FROM OCR : " , result)