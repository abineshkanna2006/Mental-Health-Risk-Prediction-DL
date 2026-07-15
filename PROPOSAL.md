# Project Proposal: Mental Health Risk Prediction using Deep Neural Networks

**Course Code & Title:** 23ADC04 — Deep Learning  
**Student Name:** [Student Name]  
**Roll Number:** [Roll Number]  
**Project Title:** Mental Health Risk Prediction using Deep Neural Networks  

---

## 1. Project Objective
The primary objective of this project is to design, implement, and deploy a robust Deep Learning architecture capable of classifying and predicting mental health risk levels (**Low Risk**, **Moderate Risk**, **High Risk**) among collegiate students. Early identification of mental health vulnerability is critical for proactive psychological counseling and academic intervention. By synthesizing multi-dimensional behavioral metrics—including sleep quality, academic pressure, screen time, physical activity, and social engagement—this model aims to capture non-linear interactions and sequential behavioral patterns that traditional linear statistical models fail to detect.

---

## 2. Dataset Source & Description
- **Dataset Title:** Student Mental Health Risk & Behavioral Indicators Dataset
- **Source:** Multi-source empirical dataset synthesized and enriched from epidemiological surveys (e.g., student stress evaluations, Kaggle mental health surveys, and clinical anxiety/depression screening inventories) (`data/student_mental_health_data.csv`).
- **Sample Size:** $N = 5,000$ validated student records (exceeding the minimum 1,000 samples checklist requirement).
- **Feature Space (14 Multi-Domain Indicators):**
  1. *Academic Factors:* `Study_Hours_Per_Day`, `Academic_Pressure` (Scale 1–10), `CGPA` (Scale 0–10), `Attendance_Rate` (Percentage).
  2. *Sleep & Lifestyle:* `Sleep_Duration_Hours`, `Sleep_Quality_Score` (Scale 1–10), `Screen_Time_Hours`, `Physical_Activity_Hours_Per_Week`, `Dietary_Habits_Score` (Scale 1–10).
  3. *Socio-Emotional Factors:* `Social_Interaction_Hours_Per_Week`, `Financial_Stress` (Scale 1–10), `Relationship_Status`, `Part_Time_Job`, `Family_History_Mental_Illness`.
- **Target Class:** `Mental_Health_Risk_Level` (`0: Low Risk`, `1: Moderate Risk`, `2: High Risk`).

---

## 3. Architecture Overview (DeepMentalHealthNet)
To satisfy the multi-paradigm requirements of the deep learning curriculum and achieve superior predictive performance on complex tabular and behavioral data, we propose **DeepMentalHealthNet**—a hybrid deep neural network combining **1D Convolutional Neural Networks (CNN)**, **Bidirectional Long Short-Term Memory (BiLSTM)** networks, and **Multi-Head Self-Attention**:
1. **Feature Embedding & Spatial Projection:** Projects standard-scaled numerical features and one-hot encoded categorical variables into a higher-dimensional dense sequence space ($B, T, D$).
2. **Module 2 Concept — 1D Convolutional Block (`Conv1D` + `BatchNorm` + `GELU`):** Extracts local feature interactions across correlated physiological clusters (e.g., interaction between `Sleep_Duration_Hours` and `Academic_Pressure`).
3. **Module 1 Concept — Bidirectional LSTM Block (`BiLSTM`):** Captures multi-directional dependencies across feature representations, modeling holistic behavioral state indicators.
4. **Module 2 Concept — Multi-Head Self-Attention Layer (`MultiHeadAttention`):** Dynamically assigns attention weights to dominant stress indicators, enabling explainability (e.g., highlighting when screen time overload and financial stress dominate risk).
5. **Classification Head:** Global pooling followed by dense layers with dropout regularization and a `Softmax` activation predicting probability over the 3 risk classes.

---

## 4. Expected Outcome & Deliverables
- **High-Accuracy Classification:** Achieving $\ge 88\%$ multi-class accuracy and high macro F1-score across all three risk categories.
- **Explainability & Interpretability:** Extraction of attention weights and feature impacts to provide clear clinical and personal rationale for every prediction.
- **Production Deployment via Streamlit:** A user-friendly, responsive web application (`app.py`) allowing students and institutional counselors to input risk variables and receive real-time risk assessment, visual probability distributions, and personalized wellness recommendations.
