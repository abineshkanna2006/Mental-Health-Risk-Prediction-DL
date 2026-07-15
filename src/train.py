"""
Model Training & Evaluation Pipeline (`src/train.py`)
Trains `DeepMentalHealthNet` (Conv1D + BiLSTM + Attention + MLP) on student mental health risk data.
Logs evaluation metrics (Accuracy, F1-score, AUC, Confusion Matrix) and saves:
- Best model weights: `models/best_model.keras` & `models/best_model.h5`
- Training/Validation curves: `assets/training_curves.png`
- Confusion matrix plot: `assets/confusion_matrix.png`
- Multi-class ROC curves: `assets/roc_curves.png`
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model import build_deep_mental_health_net

def plot_training_curves(history, output_path="assets/training_curves.png"):
    """Plots and saves epoch-wise training and validation loss/accuracy curves."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Loss Curve
    ax1.plot(history.history["loss"], label="Train Loss", color="#0284c7", linewidth=2.5)
    ax1.plot(history.history["val_loss"], label="Validation Loss", color="#e11d48", linewidth=2.5, linestyle="--")
    ax1.set_title("Model Loss Across Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Categorical Crossentropy Loss", fontsize=11)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Accuracy Curve
    ax1_acc = ax2
    ax1_acc.plot(history.history["accuracy"], label="Train Accuracy", color="#059669", linewidth=2.5)
    ax1_acc.plot(history.history["val_accuracy"], label="Validation Accuracy", color="#d97706", linewidth=2.5, linestyle="--")
    ax1_acc.set_title("Model Accuracy Across Epochs", fontsize=13, fontweight="bold")
    ax1_acc.set_xlabel("Epoch", fontsize=11)
    ax1_acc.set_ylabel("Accuracy", fontsize=11)
    ax1_acc.legend(loc="lower right")
    ax1_acc.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Training curves saved to '{output_path}'")

def plot_confusion_matrix(y_true, y_pred, output_path="assets/confusion_matrix.png"):
    """Plots and saves an annotated confusion matrix."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Low Risk (0)", "Moderate Risk (1)", "High Risk (2)"]
    
    plt.figure(figsize=(8, 6.5), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 13, "weight": "bold"})
    plt.title("Confusion Matrix — DeepMentalHealthNet Test Evaluation", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Risk Level", fontsize=11, fontweight="bold")
    plt.ylabel("True Risk Level", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Confusion matrix saved to '{output_path}'")

def plot_roc_curves(y_true, y_probs, output_path="assets/roc_curves.png"):
    """Plots One-vs-Rest ROC curves and computes AUC for each class."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    labels = ["Low Risk (0)", "Moderate Risk (1)", "High Risk (2)"]
    colors = ["#0284c7", "#d97706", "#e11d48"]
    
    plt.figure(figsize=(9, 7), dpi=300)
    
    auc_scores = {}
    for i in range(3):
        y_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_probs[:, i])
        auc_val = roc_auc_score(y_binary, y_probs[:, i])
        auc_scores[labels[i]] = auc_val
        plt.plot(fpr, tpr, color=colors[i], lw=2.5, label=f"{labels[i]} (AUC = {auc_val:.3f})")
        
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("Multi-Class ROC Curves — DeepMentalHealthNet", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True, facecolor="#f8fafc")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  ROC curves saved to '{output_path}'")
    return auc_scores

def train_and_evaluate(data_dir: str = "data/processed", models_dir: str = "models"):
    """Main training loop."""
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Loading preprocessed datasets from '{data_dir}'...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    class_weights = joblib.load(os.path.join(data_dir, "class_weights.pkl"))
    
    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    
    model = build_deep_mental_health_net(input_dim=X_train.shape[1], num_classes=3, learning_rate=0.0015)
    
    os.makedirs(models_dir, exist_ok=True)
    best_keras_path = os.path.join(models_dir, "best_model.keras")
    best_h5_path = os.path.join(models_dir, "best_model.h5")
    
    callbacks = [
        ModelCheckpoint(filepath=best_keras_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        ModelCheckpoint(filepath=best_h5_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
    ]
    
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Starting model training across epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n[{pd.Timestamp.now().strftime('%H:%M:%S')}] Generating plots and evaluating on Test Set...")
    plot_training_curves(history)
    
    # Predict on test set
    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)
    
    plot_confusion_matrix(y_test, y_pred)
    auc_scores = plot_roc_curves(y_test, y_probs)
    
    # Calculate overall metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    macro_prec = precision_score(y_test, y_pred, average="macro")
    macro_rec = recall_score(y_test, y_pred, average="macro")
    macro_auc = roc_auc_score(y_test, y_probs, multi_class="ovr", average="macro")
    
    print("\n" + "="*70)
    print("                 FINAL EVALUATION RESULTS (TEST SET)")
    print("="*70)
    print(f"  Overall Accuracy : {acc*100:.2f}%")
    print(f"  Macro Precision  : {macro_prec:.4f}")
    print(f"  Macro Recall     : {macro_rec:.4f}")
    print(f"  Macro F1-Score   : {macro_f1:.4f}")
    print(f"  Macro ROC-AUC    : {macro_auc:.4f}")
    print("-" * 70)
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low Risk (0)", "Moderate Risk (1)", "High Risk (2)"]))
    print("="*70)
    
    # Save final test evaluation summary dictionary for notebooks/report
    eval_summary = {
        "accuracy": float(acc),
        "precision": float(macro_prec),
        "recall": float(macro_rec),
        "f1_score": float(macro_f1),
        "roc_auc": float(macro_auc),
        "class_aucs": auc_scores
    }
    joblib.dump(eval_summary, os.path.join(models_dir, "eval_metrics.pkl"))
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] Best weights saved to '{best_keras_path}' & '{best_h5_path}'")
    return model, eval_summary

if __name__ == "__main__":
    train_and_evaluate()
