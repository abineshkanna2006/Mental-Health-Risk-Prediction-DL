"""
Data Preprocessing Pipeline (`src/preprocess.py`)
Handles data cleaning, normalization (`StandardScaler`), categorical encoding (`OneHotEncoder`),
stratified train/validation/test splitting, and class weight calculation.
Saves preprocessing artifacts (`models/scaler.pkl`) for real-time inference in Streamlit.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight

def get_feature_definitions():
    """Returns numerical and categorical feature column lists."""
    num_cols = [
        "Age", "Study_Hours_Per_Day", "Academic_Pressure", "CGPA", 
        "Attendance_Rate", "Sleep_Duration_Hours", "Sleep_Quality_Score", 
        "Screen_Time_Hours", "Physical_Activity_Hours_Per_Week", 
        "Dietary_Habits_Score", "Social_Interaction_Hours_Per_Week", "Financial_Stress"
    ]
    cat_cols = [
        "Gender", "Degree_Level", "Relationship_Status", 
        "Part_Time_Job", "Family_History_Mental_Illness"
    ]
    return num_cols, cat_cols

def load_and_preprocess_data(data_path: str = "data/student_mental_health_data.csv", 
                             save_artifacts: bool = True,
                             output_dir: str = "data/processed",
                             models_dir: str = "models"):
    """
    Loads raw CSV, transforms features, splits into Train/Val/Test (70/15/15),
    computes class weights, and saves preprocessor objects.
    """
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Loading raw data from '{data_path}'...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run python src/get_dataset.py first.")
        
    df = pd.read_csv(data_path)
    
    # Check for missing values and handle if any
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"  Found {missing_count} missing values. Filling numerical with median and categorical with mode...")
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    
    num_cols, cat_cols = get_feature_definitions()
    
    X = df[num_cols + cat_cols]
    y = df["Mental_Health_Risk_Level"].values
    
    # Build ColumnTransformer with StandardScaler and OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols)
        ]
    )
    
    print("  Fitting preprocessor and transforming features...")
    X_processed = preprocessor.fit_transform(X)
    
    # Extract feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_feature_names
    print(f"  Total transformed feature dimensions: {len(all_feature_names)} ({len(num_cols)} numerical + {len(cat_feature_names)} categorical encoded)")
    
    # Stratified Train/Val/Test Split (70% Train, 15% Val, 15% Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_processed, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"  Split sizes -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Compute class weights to handle mild/moderate class imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"  Computed Class Weights: {class_weight_dict}")
    
    if save_artifacts:
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save preprocessor and feature names
        joblib.dump({
            "preprocessor": preprocessor,
            "feature_names": all_feature_names,
            "num_cols": num_cols,
            "cat_cols": cat_cols
        }, os.path.join(models_dir, "scaler.pkl"))
        
        # Save processed data matrices
        np.save(os.path.join(output_dir, "X_train.npy"), X_train)
        np.save(os.path.join(output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(output_dir, "X_val.npy"), X_val)
        np.save(os.path.join(output_dir, "y_val.npy"), y_val)
        np.save(os.path.join(output_dir, "X_test.npy"), X_test)
        np.save(os.path.join(output_dir, "y_test.npy"), y_test)
        joblib.dump(class_weight_dict, os.path.join(output_dir, "class_weights.pkl"))
        
        print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Preprocessing artifacts saved to '{models_dir}/scaler.pkl' and '{output_dir}/'")
        
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), preprocessor, class_weight_dict, all_feature_names

if __name__ == "__main__":
    load_and_preprocess_data()
