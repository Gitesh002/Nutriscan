import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tempfile import NamedTemporaryFile
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Import custom modules (assuming they're in the same directory)
import sys
sys.path.append(".")
try:
    import OcrToTableTool as ottt
    import TableExtractor as te
    import TableLinesRemover as tlr
except ImportError:
    st.error("Unable to import OCR modules. Make sure they are in the same directory.")

# Define functions for the app
def load_model_and_scaler():
    """Load the trained model and scaler from disk or train them if not available"""
    try:
        # Try to load pre-saved model and scaler
        model = pickle.load(open('nutrition_model.pkl', 'rb'))
        scaler = pickle.load(open('nutrition_scaler.pkl', 'rb'))
        return model, scaler
    except FileNotFoundError:
        # If not found, train new ones
        st.info("Training new model (first-time setup)...")
        return train_nutrition_model()

def train_nutrition_model():
    """Train the nutrition prediction model"""
    try:
        # Load dataset
        data = pd.read_csv("nutrition.csv", sep="\t", nrows=10000, low_memory=False)
        
        # Selecting features and target
        data_features = data[["energy_100g", "saturated-fat_100g", "carbohydrates_100g", 
                             "fiber_100g", "fat_100g", "proteins_100g", "salt_100g"]]
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
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=500, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save model and scaler for future use
        pickle.dump(model, open('nutrition_model.pkl', 'wb'))
        pickle.dump(scaler, open('nutrition_scaler.pkl', 'wb'))
        
        return model, scaler
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None

def process_image(uploaded_file, model, scaler):
    """Process uploaded image through OCR pipeline and predict nutrition score"""
    try:
        # Save uploaded file to a temporary file
        with NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        
        # Extract table from image
        table_extractor = te.TableExtractor(temp_path)
        perspective_corrected_image = table_extractor.execute()
        
        # Remove lines from table
        lines_remover = tlr.TableLinesRemover(perspective_corrected_image)
        image_without_lines = lines_remover.execute()
        
        # Perform OCR
        ocr_tool = ottt.OcrToTableTool(image_without_lines, perspective_corrected_image)
        table_data = ocr_tool.execute()
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        # Display OCR results
        st.subheader("OCR Results")
        for row in table_data:
            st.write(row)
        
        # Process OCR data for prediction
        result = predict_nutrition_score(table_data, model, scaler)
        
        return result
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def prepare_ocr_data_for_model(ocr_table_data, scaler):
    """Transform OCR table data into a format suitable for the ML model prediction."""
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
        ". PROTEIN": "proteins_100g",
        "PROTEIN": "proteins_100g",
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
                st.warning(f"Could not convert value for {label}: {value_str}")
    
    # Create DataFrame
    df = pd.DataFrame([nutrition_dict])
    
    # Scale the data
    scaled_data = scaler.transform(df)
    
    return df, scaled_data

def predict_nutrition_score(ocr_data, model, scaler):
    """Make a prediction using OCR data"""
    # Prepare OCR data for prediction
    original_df, scaled_data = prepare_ocr_data_for_model(ocr_data, scaler)
    
    # Make prediction
    prediction = model.predict(scaled_data)
    
    return {
        "input_features": original_df.to_dict(orient='records')[0],
        "predicted_score": float(prediction[0])
    }

def process_manual_input(manual_values, model, scaler):
    """Process manually entered nutrition values and predict score"""
    try:
        # Create DataFrame from input
        df = pd.DataFrame([manual_values])
        
        # Scale data
        scaled_data = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        
        return {
            "input_features": manual_values,
            "predicted_score": float(prediction[0])
        }
    except Exception as e:
        st.error(f"Error processing manual input: {e}")
        return None

def visualize_results(result):
    """Create visualizations for the prediction results"""
    if not result:
        return
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot feature values
    features = list(result["input_features"].keys())
    values = list(result["input_features"].values())
    
    # Format feature names for better display
    display_features = [f.replace('_100g', '').replace('-', ' ').title() for f in features]
    
    # Bar chart of nutritional values
    ax1.barh(display_features, values, color='skyblue')
    ax1.set_title('Nutritional Values (per 100g)')
    ax1.set_xlabel('Amount')
    
    # Nutrition score gauge
    score = result["predicted_score"]
    
    # Define score ranges and colors
    score_ranges = [
        (-20, -1, 'green'),  # Excellent
        (0, 2, 'lightgreen'),  # Good
        (3, 10, 'yellow'),  # Average
        (11, 18, 'orange'),  # Poor
        (19, 40, 'red')  # Bad
    ]
    
    # Find color for current score
    score_color = 'gray'
    for start, end, color in score_ranges:
        if start <= score <= end:
            score_color = color
            break
    
    # Create gauge chart
    ax2.pie([1], colors=[score_color], startangle=90, wedgeprops=dict(width=0.3))
    ax2.text(0, 0, f"{score:.1f}", ha='center', va='center', fontsize=24, fontweight='bold')
    ax2.set_title('Nutrition Score')
    
    # Add score scale
    scale_pos = np.linspace(-0.9, 0.9, 5)
    scale_labels = ["0", "10", "20+"]
    for pos, label in zip(scale_pos, scale_labels):
        ax2.text(pos, -1.2, label, ha='center')
    
    plt.tight_layout()
    return fig

# Main Streamlit app
def main():
    st.set_page_config(page_title="Nutrition Score Predictor", layout="wide")
    
    st.title("Nutrition Score Prediction App")
    st.write("""
    This app predicts the nutrition score of food products based on their nutritional information.
    You can either upload an image of a nutrition label for automatic extraction or manually enter the values.
    """)
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.error("Failed to load or train model. Please check your data files.")
        return
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["OCR from Image", "Manual Entry"])
    
    with tab1:
        st.header("Upload Nutrition Label Image")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Process Image"):
                with st.spinner("Processing image..."):
                    result = process_image(uploaded_file, model, scaler)
                
                if result:
                    st.success("Image processed successfully!")
                    
                    # Display prediction result
                    st.subheader(f"Predicted Nutrition Score: {result['predicted_score']:.2f}")
                    
                    # Visualize results
                    fig = visualize_results(result)
                    if fig:
                        st.pyplot(fig)
    
    with tab2:
        st.header("Manual Nutrition Data Entry")
        
        # Create a form for manual input
        with st.form("manual_input_form"):
            # Create two columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                energy = st.number_input("Energy (kcal/100g)", min_value=0.0, value=0.0)
                carbs = st.number_input("Carbohydrates (g/100g)", min_value=0.0, value=0.0)
                fiber = st.number_input("Fiber (g/100g)", min_value=0.0, value=0.0)
                fat = st.number_input("Total Fat (g/100g)", min_value=0.0, value=0.0)
            
            with col2:
                saturated_fat = st.number_input("Saturated Fat (g/100g)", min_value=0.0, value=0.0)
                proteins = st.number_input("Proteins (g/100g)", min_value=0.0, value=0.0)
                salt = st.number_input("Salt (g/100g)", min_value=0.0, value=0.0)
            
            # Submit button
            submitted = st.form_submit_button("Predict Score")
        
        if submitted:
            # Collect inputs into a dictionary
            manual_values = {
                "energy_100g": energy,
                "saturated-fat_100g": saturated_fat,
                "carbohydrates_100g": carbs,
                "fiber_100g": fiber,
                "fat_100g": fat,
                "proteins_100g": proteins,
                "salt_100g": salt
            }
            
            # Process manual input
            result = process_manual_input(manual_values, model, scaler)
            
            if result:
                # Display prediction result
                st.subheader(f"Predicted Nutrition Score: {result['predicted_score']:.2f}")
                
                # Visualize results
                fig = visualize_results(result)
                if fig:
                    st.pyplot(fig)
    
    # Add explanation about nutrition scores
    with st.expander("About Nutrition Scores"):
        st.write("""
        ## Understanding Nutrition Scores
        
        The nutrition score is based on the French Nutri-Score system, which rates food from -15 (healthiest) to +40 (least healthy).
        
        ### Score Ranges:
        - **-15 to -1**: Excellent nutritional quality (Green)
        - **0 to 2**: Good nutritional quality (Light Green)
        - **3 to 10**: Average nutritional quality (Yellow)
        - **11 to 18**: Poor nutritional quality (Orange)
        - **19 and above**: Bad nutritional quality (Red)
        
        The score considers both beneficial nutrients (fiber, protein) and nutrients to limit (saturated fat, sugar, salt).
        """)

if __name__ == "__main__":
    main()