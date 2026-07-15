"""
Deep Neural Network Architecture (`src/model.py`)
Implements `DeepMentalHealthNet` — a hybrid Deep Learning model satisfying course rubric:
- Module 1 Concepts: Bidirectional LSTM (`BiLSTM`) & Multi-Layer Perceptron (`MLP`)
- Module 2 Concepts: 1D Convolutional Neural Network (`Conv1D`) & Multi-Head Self-Attention (`MultiHeadAttention`)
Also includes automated architecture block diagram generation (`docs/architecture.png`).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_deep_mental_health_net(input_dim: int, num_classes: int = 3, learning_rate: float = 0.001) -> models.Model:
    """
    Constructs and compiles the hybrid DeepMentalHealthNet (Conv1D + BiLSTM + Attention + MLP).
    
    Layer-by-Layer Architecture:
    1. Input Projection & Sequence Reshaping: Transforms raw tabular vector into sequence tensor (B, T, E).
    2. Conv1D Block (Module 2): Extracts local non-linear interactions across feature clusters.
    3. BiLSTM Block (Module 1): Models multi-directional dependencies across feature representations.
    4. Multi-Head Self-Attention Block (Module 2): Dynamically attends to high-risk behavioral triggers.
    5. Global Pooling & Dense Classification Head: Outputs probability over 3 risk classes.
    """
    inputs = layers.Input(shape=(input_dim,), name="tabular_input")
    
    # 1. Feature Sequence Projection (Embedding tabular vector into 3D sequence: [Batch, Sequence_Len, Feature_Dim])
    # Project each feature into an embedding of dimension 32
    embedding_dim = 32
    x_proj = layers.Dense(input_dim * embedding_dim, activation="gelu", kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x_seq = layers.Reshape((input_dim, embedding_dim), name="sequence_embedding")(x_proj)
    x_seq = layers.LayerNormalization(name="embed_norm")(x_seq)
    
    # 2. Module 2 Concept — 1D Convolutional Block (Local Feature Interaction Extraction)
    conv1 = layers.Conv1D(filters=64, kernel_size=3, padding="same", name="conv1d_1")(x_seq)
    conv1 = layers.BatchNormalization(name="bn_conv1")(conv1)
    conv1 = layers.Activation("gelu")(conv1)
    conv1 = layers.Dropout(0.25, name="drop_conv1")(conv1)
    
    conv2 = layers.Conv1D(filters=64, kernel_size=3, padding="same", name="conv1d_2")(conv1)
    conv2 = layers.BatchNormalization(name="bn_conv2")(conv2)
    conv2 = layers.Activation("gelu")(conv2)
    conv_out = layers.Dropout(0.25, name="drop_conv2")(conv2)
    
    # Residual connection from sequence embedding to conv output if dimensions match (or project)
    conv_res = layers.Dense(64)(x_seq)
    x_conv = layers.Add(name="conv_residual")([conv_out, conv_res])
    
    # 3. Module 1 Concept — Bidirectional LSTM Block (Sequential & Feature Dependency Modeling)
    lstm_out = layers.Bidirectional(
        layers.LSTM(units=64, return_sequences=True, dropout=0.2, recurrent_dropout=0.0),
        name="bilstm_block"
    )(x_conv)
    
    # 4. Module 2 Concept — Multi-Head Self-Attention Block (Explainable Risk Trigger Weighting)
    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=32, name="self_attention")(query=lstm_out, value=lstm_out)
    attn_out = layers.Dropout(0.2, name="drop_attn")(attn_out)
    # Residual connection + LayerNormalization
    x_attn = layers.Add(name="attn_residual")([lstm_out, attn_out])
    x_attn = layers.LayerNormalization(name="attn_norm")(x_attn)
    
    # 5. Global Pooling & MLP Classification Head (Module 1)
    pooled = layers.GlobalAveragePooling1D(name="global_avg_pool")(x_attn)
    
    dense1 = layers.Dense(64, activation="gelu", kernel_regularizer=regularizers.l2(1e-4), name="dense_mlp_1")(pooled)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Dropout(0.35, name="drop_mlp_1")(dense1)
    
    dense2 = layers.Dense(32, activation="gelu", kernel_regularizer=regularizers.l2(1e-4), name="dense_mlp_2")(dense1)
    dense2 = layers.Dropout(0.2, name="drop_mlp_2")(dense2)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="risk_output")(dense2)
    
    # Assemble model
    model = models.Model(inputs=inputs, outputs=outputs, name="DeepMentalHealthNet")
    
    # Compile with AdamW / Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def generate_architecture_diagram(output_path: str = "docs/architecture.png"):
    """
    Generates a clean, publication-quality block diagram illustrating the layer-by-layer
    flow of DeepMentalHealthNet (Conv1D + BiLSTM + Attention + MLP).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    
    # Title
    ax.text(7, 8.1, "DeepMentalHealthNet Architecture\n(Hybrid 1D-CNN + BiLSTM + Multi-Head Self-Attention)", 
            fontsize=15, fontweight="bold", ha="center", va="center", color="#1f2937")
    
    # Layer definitions: (X, Y, Width, Height, Color, Title, Subtitle)
    blocks = [
        (0.5, 6.0, 2.6, 1.2, "#e0f2fe", "#0284c7", "Input Layer", "Tabular Features (14 dims)\nStandardScaler + OneHot"),
        (3.8, 6.0, 2.6, 1.2, "#dcfce7", "#16a34a", "Sequence Embedding", "Dense Projection -> Reshape\nSequence Tensor (B, T, 32)"),
        (7.1, 6.0, 2.8, 1.2, "#fef3c7", "#d97706", "Module 2: Conv1D Block", "2x Conv1D (64 filters) +\nBatchNorm + GELU + Dropout"),
        (10.6, 6.0, 2.8, 1.2, "#f3e8ff", "#9333ea", "Module 1: BiLSTM Block", "Bidirectional LSTM\n(64 units, return sequences)"),
        
        # Second row
        (10.6, 3.2, 2.8, 1.2, "#ffe4e6", "#e11d48", "Module 2: Self-Attention", "Multi-Head Attention (4 heads)\n+ Residual & LayerNorm"),
        (7.1, 3.2, 2.8, 1.2, "#e0e7ff", "#4f46e5", "Global Pooling", "GlobalAveragePooling1D\nSpatial-Temporal Collapse"),
        (3.8, 3.2, 2.6, 1.2, "#ffedd5", "#ea580c", "Module 1: MLP Head", "Dense (64) -> Dropout ->\nDense (32) -> GELU"),
        (0.5, 3.2, 2.6, 1.2, "#ecfdf5", "#059669", "Output Prediction", "3-Class Softmax\n(Low, Moderate, High Risk)")
    ]
    
    for x, y, w, h, bg_color, border_color, title, subtitle in blocks:
        # Draw rounded rectangle block
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            linewidth=2, edgecolor=border_color, facecolor=bg_color
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, fontsize=11, fontweight="bold", ha="center", va="center", color="#111827")
        ax.text(x + w/2, y + h*0.28, subtitle, fontsize=8.5, ha="center", va="center", color="#374151")
        
    # Draw Arrows connecting blocks
    arrows = [
        ((3.1, 6.6), (3.8, 6.6)),   # Input -> Embedding
        ((6.4, 6.6), (7.1, 6.6)),   # Embedding -> Conv1D
        ((9.9, 6.6), (10.6, 6.6)),  # Conv1D -> BiLSTM
        ((12.0, 6.0), (12.0, 4.4)), # BiLSTM -> Attention (Vertical down)
        ((10.6, 3.8), (9.9, 3.8)),  # Attention -> Pooling (Left)
        ((7.1, 3.8), (6.4, 3.8)),   # Pooling -> MLP (Left)
        ((3.8, 3.8), (3.1, 3.8))    # MLP -> Output (Left)
    ]
    
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=2.5, color="#4b5563"))
        
    # Add Legend/Footer note
    ax.text(7, 1.4, "Module 1 Concepts: Bidirectional LSTM (BiLSTM) + Multi-Layer Perceptron (MLP)\n"
                    "Module 2 Concepts: 1D Convolutional Block (Conv1D) + Multi-Head Self-Attention Mechanism",
            fontsize=10.5, ha="center", va="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.5))
            
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Architecture block diagram saved to '{output_path}'")

if __name__ == "__main__":
    generate_architecture_diagram()
    # Test building model with dummy dim
    model = build_deep_mental_health_net(input_dim=25)
    model.summary()
