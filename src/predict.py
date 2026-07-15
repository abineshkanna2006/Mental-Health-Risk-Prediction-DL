"""
Real-Time Inference & Risk Explanation Module (`src/predict.py`)
Loads trained model (`models/best_model.keras`) and preprocessing pipeline (`models/scaler.pkl`)
to predict student mental health risk levels (`Low`, `Moderate`, `High`) and extract top risk factor explanations.
"""

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# Suppress verbose TF warnings during inference
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def load_inference_artifacts(model_path: str = "models/best_model.keras", 
                             scaler_path: str = "models/scaler.pkl"):
    """Loads fitted preprocessor and trained Keras model."""
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at '{scaler_path}'. Please run python src/preprocess.py first.")
    if not os.path.exists(model_path):
        # Fallback to .h5 if .keras not found
        h5_path = model_path.replace(".keras", ".h5")
        if os.path.exists(h5_path):
            model_path = h5_path
        else:
            raise FileNotFoundError(f"Trained model not found at '{model_path}'. Please run python src/train.py first.")
            
    scaler_data = joblib.load(scaler_path)
    preprocessor = scaler_data["preprocessor"]
    model = tf.keras.models.load_model(model_path, compile=False)
    return model, preprocessor, scaler_data

def analyze_top_risk_factors(student_dict: dict) -> list:
    """
    Analyzes specific student input values against epidemiological clinical thresholds
    to explain the top personalized drivers of mental health vulnerability.
    """
    factors = []
    
    # Check physiological & sleep indicators
    if student_dict.get("Sleep_Duration_Hours", 7) <= 5.5:
        factors.append(f"Severe Sleep Deprivation ({student_dict['Sleep_Duration_Hours']} hrs/day)")
    elif student_dict.get("Sleep_Quality_Score", 7) <= 4:
        factors.append(f"Poor Sleep Quality Score ({student_dict['Sleep_Quality_Score']}/10)")
        
    # Check academic pressure & study load
    if student_dict.get("Academic_Pressure", 5) >= 8:
        factors.append(f"High Academic Pressure ({student_dict['Academic_Pressure']}/10)")
    if student_dict.get("Study_Hours_Per_Day", 5) >= 10.0:
        factors.append(f"Excessive Study Hours ({student_dict['Study_Hours_Per_Day']} hrs/day)")
        
    # Check screen time & physical activity
    if student_dict.get("Screen_Time_Hours", 5) >= 10.0:
        factors.append(f"Heavy Screen Time Overload ({student_dict['Screen_Time_Hours']} hrs/day)")
    if student_dict.get("Physical_Activity_Hours_Per_Week", 3) <= 1.0:
        factors.append(f"Sedentary Lifestyle ({student_dict['Physical_Activity_Hours_Per_Week']} hrs physical activity/wk)")
        
    # Check socio-emotional & economic indicators
    if student_dict.get("Financial_Stress", 5) >= 8:
        factors.append(f"High Financial Stress ({student_dict['Financial_Stress']}/10)")
    if student_dict.get("Relationship_Status") == "Complicated/Breakup":
        factors.append("Relationship Distress / Breakup")
    if student_dict.get("Family_History_Mental_Illness") == "Yes":
        factors.append("Family History of Mental Illness")
    if student_dict.get("Social_Interaction_Hours_Per_Week", 10) <= 2.0:
        factors.append(f"Social Isolation ({student_dict['Social_Interaction_Hours_Per_Week']} hrs social interaction/wk)")
    if student_dict.get("Attendance_Rate", 85) < 70.0:
        factors.append(f"Low Academic Attendance ({student_dict['Attendance_Rate']}%)")
        
    if not factors:
        factors.append("No critical individual risk spikes detected; general wellness profile.")
        
    return factors[:4] # Return top 4 explanatory drivers

def predict_student_risk(student_data: dict, 
                         model_path: str = "models/best_model.keras", 
                         scaler_path: str = "models/scaler.pkl") -> dict:
    """
    Main inference function. Converts single dictionary or DataFrame row into transformed vector,
    runs prediction, and returns structured result with class probabilities and clinical explanation.
    """
    model, preprocessor, scaler_data = load_inference_artifacts(model_path, scaler_path)
    
    # Ensure DataFrame structure matching preprocessor expectations
    num_cols = scaler_data["num_cols"]
    cat_cols = scaler_data["cat_cols"]
    
    # Create single-row DataFrame
    df_input = pd.DataFrame([student_data])
    
    # Check for missing columns and fill with reasonable default if any
    for col in num_cols:
        if col not in df_input.columns:
            df_input[col] = 5.0
    for col in cat_cols:
        if col not in df_input.columns:
            df_input[col] = "Undergraduate" if col == "Degree_Level" else "No"
            
    # Transform
    X_processed = preprocessor.transform(df_input[num_cols + cat_cols])
    
    # Predict
    probs = model.predict(X_processed, verbose=0)[0]
    pred_class = int(np.argmax(probs))
    
    labels_map = {
        0: "Low Risk",
        1: "Moderate Risk",
        2: "High Risk"
    }
    
    prob_dict = {
        "Low Risk": float(probs[0] * 100),
        "Moderate Risk": float(probs[1] * 100),
        "High Risk": float(probs[2] * 100)
    }
    
    top_factors = analyze_top_risk_factors(student_data)
    
    return {
        "predicted_class": pred_class,
        "predicted_label": labels_map[pred_class],
        "confidence_percentage": float(probs[pred_class] * 100),
        "probability_distribution": prob_dict,
        "top_risk_factors": top_factors
    }

if __name__ == "__main__":
    # Test sample prediction with high stress indicators
    sample_student = {
        "Age": 21,
        "Gender": "Female",
        "Degree_Level": "Undergraduate",
        "Study_Hours_Per_Day": 11.0,
        "Academic_Pressure": 9,
        "CGPA": 7.8,
        "Attendance_Rate": 72.0,
        "Sleep_Duration_Hours": 4.5,
        "Sleep_Quality_Score": 3,
        "Screen_Time_Hours": 11.5,
        "Physical_Activity_Hours_Per_Week": 0.5,
        "Dietary_Habits_Score": 4,
        "Social_Interaction_Hours_Per_Week": 1.5,
        "Financial_Stress": 8,
        "Relationship_Status": "Complicated/Breakup",
        "Part_Time_Job": "Yes",
        "Family_History_Mental_Illness": "Yes"
    }
    print("Testing inference on sample high-stress student profile...")
    try:
        res = predict_student_risk(sample_student)
        print("Prediction Result:")
        print(f"  Risk Level: {res['predicted_label']} ({res['confidence_percentage']:.1f}% confidence)")
        print(f"  Probabilities: {res['probability_distribution']}")
        print(f"  Top Risk Factors: {res['top_risk_factors']}")
    except Exception as e:
        print(f"Inference test skipped/waiting for model training: {e}")
