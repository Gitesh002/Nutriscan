Sure, here’s a professional and comprehensive `README.md` file for your GitHub repository based on your project report:

---

# NutriScan 🍎  
**ML-Driven Nutritional Analysis System Using OCR**

NutriScan is an intelligent system that integrates Optical Character Recognition (OCR) and Machine Learning to extract and analyze nutritional information from food packaging images. This tool empowers users to make healthier dietary choices through real-time nutritional assessment.

## 🚀 Features
- 📷 **OCR-based Text Extraction**: Automatically extracts nutritional facts from food label images using Tesseract and CNN-enhanced image preprocessing.
- 🧠 **ML Nutritional Scoring**: Predicts a standardized health score (1–25 scale) based on key nutrients using models like Random Forest and SVM.
- 📊 **Intuitive Visualization**: Graphically presents nutritional values and scoring insights.
- 🧪 **Manual Entry Support**: Allows users to input data if image processing isn't feasible.
- 🧬 **Personalized Recommendations**: Offers actionable health suggestions based on extracted nutrition info.

## 🛠️ Tech Stack
- **Frontend/UI**: Streamlit
- **Backend**: Python
- **OCR**: Tesseract OCR, EAST Text Detector, Custom CNN post-processing
- **Machine Learning**: Scikit-learn (Random Forest, Lasso, SVM, Decision Tree)
- **Data Processing**: Pandas, NumPy, OpenCV
- **Visualization**: Matplotlib

## 📷 OCR Workflow
1. Upload an image of the food label.
2. Preprocessing (noise reduction, contrast enhancement).
3. Text extraction using OCR.
4. Nutritional fields parsed and mapped.
5. Data sent to the ML model for score prediction.

## 📈 Machine Learning
Trained on OpenFoodFacts data:
- **Input Features**: Calories, Fats, Proteins, Sugars, Fiber, Salt, Carbs, etc.
- **Model Used**: `RandomForestRegressor` (best performance with R² = 0.95)
- **Target**: Nutrition Score (1–25 scale)

## 🧪 Sample Usage
```bash
# Clone repository
git clone https://github.com/yourusername/NutriScan.git
cd NutriScan

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

## 📂 Folder Structure
```
NutriScan/
├── app.py                 # Main Streamlit app
├── model/                 # Trained ML model and scaler
├── ocr_modules/           # Custom OCR preprocessing and table extraction scripts
├── data/                  # Dataset used for training
├── utils/                 # Helper functions
├── requirements.txt
└── README.md
```

## 🎯 Objectives
- Eliminate manual nutrition tracking
- Deliver accurate health scores from food images
- Support chronic condition management via diet
- Enhance food label accessibility

## 📌 Limitations
- OCR accuracy may drop on low-quality images
- Works best with English labels and standardized tables
- Requires internet and processing time on lower-end devices

## 🌱 Future Enhancements
- Mobile app with offline OCR support
- Multilingual label recognition
- Integration with health apps (e.g., Fitbit)
- Allergen and ingredient safety warnings

## 👥 Team
- Gitesh Anand (Davo)  
- Lakshay Chaudhary  
- Prafful Bisht  
**Guide:** Mr. Kuwar Pratap Singh

---

Let me know if you want me to convert this into a downloadable `README.md` file or customize it for a Streamlit sharing link.
