"""
Streamlit Web Application (`app.py`)
Interactive, aesthetically premium portal for Student Mental Health Risk Prediction
using `DeepMentalHealthNet` (Hybrid Conv1D + BiLSTM + Multi-Head Self-Attention).
Satisfies Phase 9/10 deployment & M1/M2/M3 curriculum integration requirements.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Ensure root and src directories are accessible for module imports
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from src.predict import predict_student_risk
except ImportError:
    from predict import predict_student_risk

# Set Streamlit Page Config
st.set_page_config(
    page_title="DeepMentalHealthNet | Student Risk Assessment Portal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphic / Modern Aesthetics
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header Gradient */
.hero-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #8b5cf6 100%);
    padding: 2.2rem;
    border-radius: 16px;
    color: white;
    box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.4);
    margin-bottom: 2rem;
}
.hero-header h1 {
    color: white;
    font-weight: 700;
    margin-bottom: 0.5rem;
    font-size: 2.3rem;
}
.hero-header p {
    color: #e0f2fe;
    font-size: 1.05rem;
    margin: 0;
}

/* Risk Status Cards */
.risk-card-low {
    background: linear-gradient(135deg, #065f46 0%, #10b981 100%);
    color: white;
    padding: 1.8rem;
    border-radius: 14px;
    box-shadow: 0 8px 20px -4px rgba(16, 185, 129, 0.4);
    text-align: center;
}
.risk-card-mod {
    background: linear-gradient(135deg, #92400e 0%, #f59e0b 100%);
    color: white;
    padding: 1.8rem;
    border-radius: 14px;
    box-shadow: 0 8px 20px -4px rgba(245, 158, 11, 0.4);
    text-align: center;
}
.risk-card-high {
    background: linear-gradient(135deg, #991b1b 0%, #ef4444 100%);
    color: white;
    padding: 1.8rem;
    border-radius: 14px;
    box-shadow: 0 8px 20px -4px rgba(239, 68, 68, 0.4);
    text-align: center;
}
.risk-title {
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0.4rem 0;
}
.risk-subtitle {
    font-size: 1rem;
    opacity: 0.9;
}

/* Explanatory & Recommendation Cards */
.section-card {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    color: #0f172a !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
}
.section-card *, .section-card b, .section-card p, .section-card span, .section-card div {
    color: #0f172a !important;
}
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="hero-header">
    <h1>🧠 DeepMentalHealthNet Assessment Portal</h1>
    <p>Hybrid 1D-CNN + Bidirectional LSTM + Multi-Head Self-Attention Architecture | Course: 23ADC04 Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Input Controls
st.sidebar.markdown("### 📋 Student Profile Assessment")
st.sidebar.markdown("Adjust sliders below to simulate or input student behavioral indicators:")

with st.sidebar.expander("🎓 1. Academic Profile", expanded=True):
    age = st.slider("Age", 17, 30, 21)
    gender = st.selectbox("Gender", ["Female", "Male", "Non-Binary/Other"])
    degree_level = st.selectbox("Degree Level", ["Undergraduate", "Postgraduate", "PhD/Professional"])
    study_hours = st.slider("Study Hours Per Day", 0.5, 14.0, 6.5, 0.5)
    academic_pressure = st.slider("Academic Pressure (1 = Low, 10 = Extreme)", 1, 10, 6)
    cgpa = st.slider("Current CGPA (out of 10.0)", 4.0, 10.0, 7.8, 0.1)
    attendance = st.slider("Class Attendance Rate (%)", 30, 100, 85)

with st.sidebar.expander("🛌 2. Sleep & Lifestyle", expanded=True):
    sleep_duration = st.slider("Sleep Duration (Hours/Day)", 2.0, 12.0, 6.5, 0.5)
    sleep_quality = st.slider("Sleep Quality Score (1 = Poor, 10 = Excellent)", 1, 10, 6)
    screen_time = st.slider("Screen Time (Hours/Day)", 1.0, 16.0, 7.5, 0.5)
    physical_activity = st.slider("Physical Activity (Hours/Week)", 0.0, 14.0, 2.5, 0.5)
    dietary_habits = st.slider("Dietary Habits Score (1 = Poor, 10 = Excellent)", 1, 10, 6)

with st.sidebar.expander("💬 3. Socio-Emotional & Financial", expanded=True):
    social_interaction = st.slider("Social Interaction (Hours/Week)", 0.0, 25.0, 8.0, 0.5)
    financial_stress = st.slider("Financial Stress (1 = None, 10 = Severe)", 1, 10, 5)
    relationship_status = st.selectbox("Relationship Status", ["Single", "In a Relationship", "Complicated/Breakup"])
    part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])
    family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

# Assemble student input dictionary
student_data = {
    "Age": age,
    "Gender": gender,
    "Degree_Level": degree_level,
    "Study_Hours_Per_Day": study_hours,
    "Academic_Pressure": academic_pressure,
    "CGPA": cgpa,
    "Attendance_Rate": attendance,
    "Sleep_Duration_Hours": sleep_duration,
    "Sleep_Quality_Score": sleep_quality,
    "Screen_Time_Hours": screen_time,
    "Physical_Activity_Hours_Per_Week": physical_activity,
    "Dietary_Habits_Score": dietary_habits,
    "Social_Interaction_Hours_Per_Week": social_interaction,
    "Financial_Stress": financial_stress,
    "Relationship_Status": relationship_status,
    "Part_Time_Job": part_time_job,
    "Family_History_Mental_Illness": family_history
}

# Main Application Body
col1, col2 = st.columns([1.1, 1.3], gap="large")

with col1:
    st.markdown("### ⚡ Real-Time Risk Estimation")
    
    try:
        model_path = os.path.join("models", "best_model.keras")
        if not os.path.exists(model_path):
            model_path = os.path.join("models", "best_model.h5")
        scaler_path = os.path.join("models", "scaler.pkl")
        
        result = predict_student_risk(student_data, model_path=model_path, scaler_path=scaler_path)
        
        # Display colored status card
        pred_class = result["predicted_class"]
        pred_label = result["predicted_label"]
        conf = result["confidence_percentage"]
        
        if pred_class == 0:
            st.markdown(f"""
            <div class="risk-card-low">
                <div style="font-size:1.1rem; text-transform:uppercase; letter-spacing:1px;">Predicted Status</div>
                <div class="risk-title">🟢 {pred_label}</div>
                <div class="risk-subtitle">Confidence Level: {conf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        elif pred_class == 1:
            st.markdown(f"""
            <div class="risk-card-mod">
                <div style="font-size:1.1rem; text-transform:uppercase; letter-spacing:1px;">Predicted Status</div>
                <div class="risk-title">🟡 {pred_label}</div>
                <div class="risk-subtitle">Confidence Level: {conf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-card-high">
                <div style="font-size:1.1rem; text-transform:uppercase; letter-spacing:1px;">Predicted Status</div>
                <div class="risk-title">🔴 {pred_label}</div>
                <div class="risk-subtitle">Confidence Level: {conf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("#### 📊 Probability Distribution")
        prob_df = pd.DataFrame({
            "Risk Tier": ["Low Risk", "Moderate Risk", "High Risk"],
            "Probability (%)": [
                result["probability_distribution"]["Low Risk"],
                result["probability_distribution"]["Moderate Risk"],
                result["probability_distribution"]["High Risk"]
            ]
        }).set_index("Risk Tier")
        
        st.bar_chart(prob_df, color="#3b82f6")
        
    except Exception as e:
        st.warning(f"Unable to run real-time inference right now. Ensure `models/best_model.keras` and `models/scaler.pkl` exist. Error: {e}")
        st.info("💡 To generate models locally, run:\n```bash\npython src/preprocess.py\npython src/train.py\n```")
        result = None

with col2:
    st.markdown("### 🔍 AI Risk Explanation & Attention Insights")
    
    if result and "top_risk_factors" in result:
        st.markdown("""
        Our hybrid **DeepMentalHealthNet** (`Conv1D + BiLSTM + Multi-Head Self-Attention`) uses built-in attention gating to highlight specific behavioral triggers contributing to the predicted risk profile:
        """)
        
        for factor in result["top_risk_factors"]:
            if "No critical" in factor:
                st.success(f"✅ **{factor}**")
            else:
                st.error(f"⚠️ **{factor}**")
                
        st.markdown("---")
        st.markdown("### 💡 Actionable Interventions & Counseling Roadmap")
        
        # Build dynamically tailored recommendations based on student profile and predicted class
        recs = []
        
        # 1. Sleep Recommendations
        sleep_hrs = student_data.get("Sleep_Duration_Hours", 7.0)
        sleep_qual = student_data.get("Sleep_Quality_Score", 6)
        if sleep_hrs < 6.5 or sleep_qual <= 4:
            recs.append(f"• <b>🛌 Sleep Hygiene & Restoration:</b> You are currently getting <b>{sleep_hrs} hrs/day</b> of sleep. We recommend aiming for 7.5–8 hours by reducing evening stimulants and establishing a regular sleep schedule.")
        else:
            recs.append(f"• <b>✅ Optimal Sleep Routine:</b> Excellent work maintaining <b>{sleep_hrs} hrs/day</b> of sleep with strong sleep quality. This consistent rest is serving as a critical protective buffer for your mental well-being.")
            
        # 2. Physical Activity Recommendations
        phys_hrs = student_data.get("Physical_Activity_Hours_Per_Week", 2.5)
        if phys_hrs <= 2.0:
            recs.append(f"• <b>🏃 Active Movement & Exercise:</b> Your physical activity is low (<b>{phys_hrs} hrs/wk</b>). Adding just 2–3 hours weekly of brisk walking, gym workouts, or campus intramurals significantly boosts endorphin levels and mental resilience.")
        else:
            recs.append(f"• <b>💪 Healthy Physical Activity:</b> Your physical activity (<b>{phys_hrs} hrs/wk</b>) is well above the sedentary threshold, helping maintain metabolic health and stress dissipation.")
            
        # 3. Academic Pressure & Study Load Recommendations
        study_hrs = student_data.get("Study_Hours_Per_Day", 6.5)
        acad_press = student_data.get("Academic_Pressure", 5)
        if acad_press >= 7 or study_hrs >= 9.0:
            recs.append(f"• <b>📚 Workload Pacing & Support:</b> High study hours (<b>{study_hrs} hrs/day</b>) or academic pressure (<b>{acad_press}/10</b>) detected. Consider utilizing campus peer tutoring or speaking with an academic advisor to optimize study efficiency and avoid cognitive burnout.")
        else:
            recs.append(f"• <b>🎓 Balanced Academic Pacing:</b> Your study load (<b>{study_hrs} hrs/day</b>) and academic pressure are currently within healthy, sustainable limits.")
            
        # 4. Screen Time Recommendations
        screen_hrs = student_data.get("Screen_Time_Hours", 7.0)
        if screen_hrs >= 9.0:
            recs.append(f"• <b>📵 Digital Detox & Eye Rest:</b> High daily screen exposure (<b>{screen_hrs} hrs/day</b>). Try setting digital wellness boundaries, such as the 20-20-20 rule and turning off screens 1 hour before bedtime.")
            
        # 5. Clinical & Guidance Pathway based on Predicted Tier
        if result["predicted_class"] == 0:
            protocol_header = "🌟 Personalized Wellness Protocol (Low Risk Category):"
            recs.append("• <b>🛡️ Protective Buffers:</b> Continue engaging in supportive collegiate clubs and social communities to sustain overall emotional wellness.")
        elif result["predicted_class"] == 1:
            protocol_header = "🟡 Early Intervention Protocol (Moderate Risk Category):"
            recs.append("• <b>💬 Guidance & Counseling Access:</b> We recommend checking in with campus student wellness coordinators or attending stress-reduction workshops before moderate stress escalates.")
        else:
            protocol_header = "🔴 Urgent Clinical Counseling Pathway (High Risk Category):"
            recs.append("• <b>🚨 Immediate Confidential Support:</b> We strongly urge reaching out to the campus Student Health & Psychological Counseling Center. Professional clinical counseling and academic accommodation are vital steps.")
            
        recs_html = "<br/><br/>".join(recs)
        st.markdown(f"""
        <div class="section-card">
            <b style="font-size:1.15rem;">{protocol_header}</b><br/><br/>
            {recs_html}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Adjust the student profile on the sidebar to view personalized risk explanations and counseling recommendations.")

# Footer Section: Architecture Overview & Module Mapping
st.markdown("---")
with st.expander("📐 View Model Architecture & Curriculum Module Mapping (M1, M2, M3)", expanded=False):
    st.markdown("""
    #### Curriculum Mapping:
    - **Module 1 (Foundational & Recurrent Models):** Implemented via **Bidirectional LSTM (`BiLSTM`)** to capture sequence dependencies across behavioral indicators, and **Multi-Layer Perceptron (`MLP`)** for non-linear classification.
    - **Module 2 (Convolutional & Attention Models):** Implemented via two sequential **1D Convolutional blocks (`Conv1D`)** for local feature interactions and a **Multi-Head Self-Attention (`MultiHeadAttention`)** layer for dynamic risk trigger weighting.
    - **Module 3 (Real-World Applications & Ethics):** Addressed through class-imbalance mitigation (`compute_class_weight`), clinical explainability, and deployment via this interactive **Streamlit Web Application**.
    """)
    arch_img = "docs/architecture.png"
    if os.path.exists(arch_img):
        st.image(arch_img, caption="DeepMentalHealthNet Layer-by-Layer Block Diagram", use_container_width=True)
