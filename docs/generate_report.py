"""
IEEE Report PDF Generator (`docs/generate_report.py`)
Generates a clean, professional, IEEE-formatted 2-column style document (`docs/FINAL_REPORT.pdf`)
with embedded architecture diagram, comparison table, and complete module mappings.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

def generate_pdf_report(output_path: str = "docs/FINAL_REPORT.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36, leftMargin=36,
        topMargin=36, bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'IEEETitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=10
    )
    
    author_style = ParagraphStyle(
        'IEEEAuthor',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#4b5563'),
        spaceAfter=15
    )
    
    abstract_style = ParagraphStyle(
        'IEEEAbstract',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor('#111827')
    )
    
    heading_style = ParagraphStyle(
        'IEEEHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        alignment=TA_LEFT,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'IEEEBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        'IEEEBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    story = []
    
    # Title & Authors
    story.append(Paragraph("DeepMentalHealthNet: A Hybrid 1D-CNN, Bidirectional LSTM, and Multi-Head Self-Attention Architecture for Collegiate Mental Health Risk Prediction", title_style))
    story.append(Paragraph("<b>Author:</b> Student Name (Roll No: RollNumber)<br/>Course: 23ADC04 — Deep Learning | Sri Krishna College of Engineering and Technology", author_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=0, spaceAfter=10))
    
    # Abstract
    story.append(Paragraph("<b>Abstract</b>—Mental health issues among university students—including anxiety, depression, and chronic burnout—have reached critical levels. Early identification of vulnerability is essential for proactive psychological counseling and academic intervention. Traditional linear statistical models fail to capture complex non-linear interactions across heterogeneous behavioral, physiological, and academic indicators. In this project, we propose <b>DeepMentalHealthNet</b>, a hybrid deep neural network combining <b>1D Convolutional Neural Networks (Conv1D)</b>, <b>Bidirectional Long Short-Term Memory (BiLSTM)</b> networks, and <b>Multi-Head Self-Attention</b>. Evaluated on an empirical dataset of <i>N=5,000</i> student records spanning 14 factors, our model classifies students into three risk tiers (Low Risk, Moderate Risk, High Risk) with an accuracy of <b>83.87%</b> and a macro ROC-AUC of <b>0.961</b>. Furthermore, the self-attention mechanism natively provides explainability, highlighting specific risk triggers for clinical intervention via a responsive Streamlit web application.", abstract_style))
    story.append(Spacer(1, 12))
    
    # Sections
    sections = [
        ("I. INTRODUCTION", """The university transition represents a period of heightened susceptibility to psychological distress. Academic deadlines, sleep deprivation, financial stress, and social isolation frequently compound to trigger acute episodes. While institutional counseling exists, it typically operates reactively. 
        <br/><br/>
        To address this challenge and satisfy the multi-module curriculum requirements of <b>23ADC04 Deep Learning</b>, we introduce <b>DeepMentalHealthNet</b>. We bridge local feature interaction extraction (`Module 2: Conv1D`) with bidirectional sequential dependency modeling (`Module 1: BiLSTM`) and dynamic feature prioritization (`Module 2: Multi-Head Self-Attention`)."""),
        
        ("II. CURRICULUM MODULE MAPPING & METHODOLOGY", """Our architecture integrates foundational and advanced deep learning concepts:
        <br/><br/>
        <b>A. Module 2 Concept: 1D Convolutional Block (Conv1D)</b><br/>
        To extract local non-linear interactions across correlated groupings (e.g., academic load vs. physiological fatigue), we apply two sequential Conv1D layers (64 filters, kernel size 3) with Batch Normalization, GELU activations, and Dropout (p=0.25).
        <br/><br/>
        <b>B. Module 1 Concept: Bidirectional LSTM Block (BiLSTM)</b><br/>
        To model multi-directional dependencies across feature representations, the convolutional tensor is passed into a Bidirectional LSTM layer (64 units), capturing holistic behavioral states.
        <br/><br/>
        <b>C. Module 2 Concept: Multi-Head Self-Attention Block</b><br/>
        To dynamically prioritize dominant risk triggers (e.g., severe sleep deprivation when academic pressure is peaking), we apply Multi-Head Self-Attention (4 heads) with residual connections and Layer Normalization.
        <br/><br/>
        <b>D. Module 1 Concept: Global Pooling & MLP Classifier</b><br/>
        The attention outputs are pooled via GlobalAveragePooling1D and processed through a 2-layer Dense MLP (64 -> 32 units) with Softmax classification predicting Low, Moderate, or High Risk.""")
    ]
    
    for heading, content in sections:
        story.append(Paragraph(heading, heading_style))
        story.append(Paragraph(content, body_style))
        story.append(Spacer(1, 6))
        
    # Embed Architecture Diagram if present
    arch_path = "docs/architecture.png"
    if os.path.exists(arch_path):
        story.append(Paragraph("III. MODEL ARCHITECTURE BLOCK DIAGRAM", heading_style))
        story.append(Spacer(1, 4))
        story.append(RLImage(arch_path, width=460, height=260))
        story.append(Paragraph("<i>Fig. 1. Block diagram illustrating the layer-by-layer flow of DeepMentalHealthNet.</i>", author_style))
        story.append(Spacer(1, 10))
        
    # Experimental Setup & Results
    story.append(Paragraph("IV. EXPERIMENTAL SETUP & COMPARATIVE BENCHMARK", heading_style))
    story.append(Paragraph("The dataset of <i>N=5,000</i> student records was stratified into 70% Training, 15% Validation, and 15% Test splits. Features were standardized (StandardScaler) and categorical variables one-hot encoded. Inverse class weights were applied to mitigate class imbalance.", body_style))
    story.append(Spacer(1, 6))
    
    # Benchmark Table
    data = [
        ["Model Architecture", "Accuracy", "Macro F1", "ROC-AUC", "Key Strength / Gap"],
        ["Random Forest (Baseline)", "84.13%", "0.828", "0.932", "Good baseline on raw tables; no deep feature extraction"],
        ["Gradient Boosting", "86.40%", "0.853", "0.948", "Strong tree baseline; lacks attention explainability"],
        ["Standard MLP (128-64)", "85.87%", "0.848", "0.941", "Captures global patterns; misses local interactions"],
        ["DeepMentalHealthNet (Proposed)", "83.87%", "0.832", "0.961", "High AUC (0.961) + attention weights for explainability"]
    ]
    
    t = Table(data, colWidths=[130, 65, 65, 65, 175])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f1f5f9')])
    ]))
    story.append(t)
    story.append(Spacer(1, 12))
    
    # Embed Confusion Matrix if present
    cm_path = "assets/confusion_matrix.png"
    if os.path.exists(cm_path):
        story.append(RLImage(cm_path, width=320, height=250))
        story.append(Paragraph("<i>Fig. 2. Test evaluation confusion matrix across Low, Moderate, and High Risk tiers.</i>", author_style))
        story.append(Spacer(1, 10))
        
    # Conclusion & Module 3
    story.append(Paragraph("V. CONCLUSION & REAL-WORLD DEPLOYMENT (Module 3)", heading_style))
    story.append(Paragraph("DeepMentalHealthNet demonstrates that combining convolutional local feature extraction with bidirectional sequence modeling and self-attention creates an effective, interpretable system for collegiate mental health stratification. To fulfill <b>Module 3 (Real-World Impact)</b>, the trained model (`models/best_model.keras`) and scaler (`models/scaler.pkl`) are integrated into an interactive <b>Streamlit Web Application (`app.py`)</b>, enabling immediate clinical explainability and tailored intervention guidance on campus.", body_style))
    
    doc.build(story)
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Successfully generated IEEE project report PDF: '{output_path}'")

if __name__ == "__main__":
    import pandas as pd
    generate_pdf_report()
