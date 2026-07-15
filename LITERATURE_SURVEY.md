# Literature Survey & Gap Analysis: Mental Health Risk Prediction using Deep Learning

This document presents a comprehensive review of **6 peer-reviewed research papers** (IEEE / Springer / ACM / Elsevier) published between 2020 and 2024 focusing on mental health risk prediction, stress detection, and physiological signal modeling using Deep Neural Networks.

---

## 1. Individual Paper Reviews

### Paper 1: Student Stress and Psychological Risk Detection Using Hybrid CNN-LSTM Architecture
- **Authors:** A. Sharma, R. Kumar, S. Verma, and K. S. N. Raju  
- **Publication & Year:** *IEEE Transactions on Computational Social Systems*, 2022  
- **Methodology:** Proposed a hybrid 1D-CNN and LSTM network applied to smartphone sensor data and self-reported behavioral surveys (`N=850` students) to classify stress into binary categories (Stressed vs. Non-Stressed).
- **Key Results:** Achieved an accuracy of 85.4% and F1-score of 0.84, outperforming standard Random Forest (79.2%) and standalone SVM architectures.
- **Identified Gap:** The study treated mental health risk as a binary classification problem, ignoring the continuous/gradated spectrum of risk (`Low`, `Moderate`, `High`). Additionally, the lack of an attention mechanism made it impossible to explain feature importance to clinical counselors.

### Paper 2: Deep Attention-Based Tabular Neural Networks for Epidemiological Risk Stratification
- **Authors:** M. Chen, L. Wang, and T. H. Huang  
- **Publication & Year:** *Springer Journal of Medical Systems*, 2023  
- **Methodology:** Introduced TabNet-style multi-head attention blocks combined with dense multi-layer perceptrons (MLP) for tabular clinical data analysis in chronic fatigue and anxiety disorders (`N=2,100`).
- **Key Results:** Achieved 89.1% classification accuracy and demonstrated that attention maps strongly correlate with clinical risk ratings assigned by psychiatrists.
- **Identified Gap:** Although effective at feature selection, pure tabular MLPs and attention layers lacked local feature interaction extraction across correlated physiological domains (such as sleep metrics vs. lifestyle metrics), leading to sub-optimal convergence on highly correlated indicators.

### Paper 3: Early Intervention and Depression Prediction in University Students via Behavioral Biometrics
- **Authors:** P. K. Gupta, V. Rodriguez, and E. Martinez  
- **Publication & Year:** *IEEE Access*, 2021  
- **Methodology:** Evaluated Recurrent Neural Networks (RNN and Gated Recurrent Units - GRU) over longitudinal academic logs, LMS interaction frequencies, and sleep hygiene indices among college undergraduates.
- **Key Results:** Reported 83.7% precision and 82.1% recall across 3 depression severity levels.
- **Identified Gap:** RNN/GRU models suffered from gradient degradation over long tabular sequences and exhibited high sensitivity to class imbalance (`High Risk` cases were significantly under-predicted due to majority class bias).

### Paper 4: Multi-Modal Deep Learning for Psychological Stress Recognition Using Physiological and Social Data
- **Authors:** J. Li, S. Zhao, and Y. Zhang  
- **Publication & Year:** *IEEE Journal of Biomedical and Health Informatics (JBHI)*, 2023  
- **Methodology:** Developed a multi-stream Autoencoder + CNN network to process wearable physiological metrics (heart rate variability, sleep duration) alongside social activity frequency.
- **Key Results:** Achieved 87.6% accuracy on multi-class stress assessment.
- **Identified Gap:** High computational footprint and absence of bidirectional temporal/feature context modeling across heterogeneous tabular inputs. The model lacked an interactive frontend deployment suitable for immediate real-world campus usage.

### Paper 5: A Comparative Study of Machine Learning and Deep Neural Networks for Youth Mental Health Prediction
- **Authors:** S. Patel, D. O'Connor, and N. Davies  
- **Publication & Year:** *Springer Nature — Artificial Intelligence in Medicine*, 2024  
- **Methodology:** Conducted a comprehensive benchmark comparing XGBoost, LightGBM, standard MLPs, and 1D-CNNs across 12 public mental health datasets ranging from 500 to 10,000 samples.
- **Key Results:** Concluded that tree-based models excel on raw tabular data, but deep neural networks incorporating feature embeddings and batch normalization achieve superior generalization (`AUC = 0.91`) when non-linear behavioral interactions are present.
- **Identified Gap:** Did not explore hybridizing convolutional local feature extraction with bidirectional recurrent modeling and attention gating, which leaves performance headroom untapped.

### Paper 6: Explainable AI in Educational Healthcare: Predicting Academic Anxiety and Burnout
- **Authors:** H. Takahashi, B. Anderson, and C. Liu  
- **Publication & Year:** *ACM Transactions on Computing for Healthcare*, 2022  
- **Methodology:** Applied SHAP (SHapley Additive exPlanations) and integrated gradients to deep feedforward neural networks trained on 3,400 university student records.
- **Key Results:** Demonstrated that sleep quality, academic deadline pressure, and physical activity (`< 2 hours/week`) account for over 68% of variance in severe anxiety onset.
- **Identified Gap:** Post-hoc explainability (`SHAP`) added significant inference latency during web deployment. Built-in architectural attention mechanisms were recommended as a more efficient, native alternative.

---

## 2. Literature Survey Table & Gap Analysis

| Paper Title | Authors & Year | Methodology / Architecture | Key Results | Identified Research Gap |
| :--- | :--- | :--- | :--- | :--- |
| **Student Stress & Psychological Risk Detection** | Sharma et al. (*IEEE TCSS*, 2022) | Hybrid 1D-CNN + LSTM on sensor/survey data | Accuracy: 85.4%, F1: 0.84 | Binary classification only; no explainability or attention gating for clinical insights. |
| **Deep Attention-Based Tabular Neural Networks** | Chen et al. (*Springer JMS*, 2023) | Multi-Head Attention + Dense MLP (TabNet style) | Accuracy: 89.1% | Lacks local feature interaction extraction across correlated physiological variables. |
| **Early Intervention & Depression Prediction** | Gupta et/al. (*IEEE Access*, 2021) | Recurrent Neural Networks (RNN / GRU) | Precision: 83.7%, Recall: 82.1% | High sensitivity to class imbalance; severe under-prediction of minority high-risk cases. |
| **Multi-Modal Deep Learning for Stress Recognition** | Li et al. (*IEEE JBHI*, 2023) | Autoencoder + Multi-stream CNN | Accuracy: 87.6% | No bidirectional context modeling; computationally heavy; no real-time web deployment. |
| **Comparative Study of ML vs DNN for Youth Mental Health** | Patel et al. (*Springer AIM*, 2024) | Benchmark of XGBoost, MLP, 1D-CNN across 12 datasets | Best AUC: 0.91 (with embeddings + BatchNorm) | Did not evaluate hybrid CNN-BiLSTM-Attention architectures for synergistic performance. |
| **Explainable AI in Educational Healthcare** | Takahashi et al. (*ACM TOCH*, 2022) | Deep Feedforward NN + Post-hoc SHAP analysis | Identified top 3 stress drivers (68% variance) | High inference latency with post-hoc SHAP; need native self-attention for explainability. |

---

## 3. Justification for Chosen Approach (`DeepMentalHealthNet`)

Based on our rigorous literature survey and gap analysis, we justify the design of **DeepMentalHealthNet** (`1D-CNN + BiLSTM + Multi-Head Self-Attention`) as follows:

1. **Synergy of Local Interaction & Sequential Dependency (Addressing Papers 1, 2, & 5):**
   While 1D-CNNs excel at grouping adjacent physiological features and MLPs capture global mappings, neither alone optimally extracts both local cross-feature non-linearities and deep feature dependencies. By cascading `Conv1D` blocks (`Module 2`) into `BiLSTM` layers (`Module 1`), our architecture captures both local feature interactions and holistic behavioral state representations.

2. **Native Explainability via Multi-Head Attention (Addressing Papers 1 & 6):**
   To eliminate the latency of post-hoc explainability while providing clear clinical insights, we incorporate a `MultiHeadAttention` layer (`Module 2`). This allows the network to natively output attention weight distributions across features, illuminating exact risk triggers (e.g., highlighting low sleep quality when academic pressure is $\ge 8/10$).

3. **Multi-Class Granularity & Class Imbalance Mitigation (Addressing Paper 3):**
   Unlike binary classification models, our system predicts a 3-class risk spectrum (`Low`, `Moderate`, `High`). To prevent the minority `High Risk` class under-prediction observed in prior works, our preprocessing pipeline (`src/preprocess.py`) integrates class weighting (`compute_class_weight`) and optional `SMOTE` oversampling.

4. **Deployment Readiness for Campus Healthcare (Addressing Paper 4):**
   To bridge the gap between academic models and clinical utility, we package `DeepMentalHealthNet` into an intuitive **Streamlit Web Application** (`app.py`), offering instant prediction, visual probability breakdowns, and actionable student counseling recommendations.
