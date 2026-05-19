"""
=============================================================
 Breast Cancer Classification using Deep Learning
 File   : breast_cancer_classifier.py
 Author : [Your Name]
 Date   : [Current Date]
=============================================================
 Description:
   End-to-end pipeline for binary classification of breast
   tumors (Malignant / Benign) using a Multi-Layer Perceptron
   (MLP) built with TensorFlow / Keras.

 Dataset:
   Wisconsin Breast Cancer Diagnostic dataset (data.csv).
   Columns: id, diagnosis (M/B), + 30 numeric feature cols.

 Usage:
   python breast_cancer_classifier.py
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

# ────────────────────────────────────────────────────────────────
# 1.  DATA ACQUISITION & EXPLORATORY ANALYSIS
# ────────────────────────────────────────────────────────────────

def load_and_explore(filepath: str = "data.csv") -> pd.DataFrame:
    """Load the dataset and print basic structural info."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Missing '{filepath}'. Please place the Wisconsin Breast "
            "Cancer CSV in the project root directory."
        )

    df = pd.read_csv(filepath)
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Shape      : {df.shape}")
    print(f"  Columns    : {list(df.columns)}")
    print(f"\nFirst 5 Records:\n{df.head(5)}")
    print(f"\nMissing Values Per Feature Column:\n{df.isna().sum()}")
    print(f"\nClass Distribution (Malignant=M / Benign=B):\n{df['diagnosis'].value_counts()}")
    return df


def run_eda(df: pd.DataFrame) -> None:
    """Generate exploratory visualisation plots."""
    sns.set_theme(style="ticks")

    # Pairplot — key morphological features
    key_vars = [
        "radius_mean", "texture_mean",
        "perimeter_mean", "area_mean", "smoothness_mean",
    ]
    pair = sns.pairplot(df, hue="diagnosis", palette="coolwarm", vars=key_vars)
    pair.savefig("feature_interaction_pairplot.png", dpi=150)
    plt.close()
    print("[EDA] Saved  → feature_interaction_pairplot.png")

    # Class balance bar chart
    plt.figure(figsize=(6, 4))
    sns.countplot(x="diagnosis", data=df, palette="Set2")
    plt.title("Diagnostic Distribution Breakdown")
    plt.tight_layout()
    plt.savefig("diagnosis_count_distribution.png", dpi=150)
    plt.close()
    print("[EDA] Saved  → diagnosis_count_distribution.png")


# ────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING & PREPROCESSING
# ────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Clean, encode, scale, and split the dataset.

    Returns
    -------
    x_train, x_test, y_train, y_test : np.ndarray
    """
    # Drop completely empty columns
    df = df.dropna(axis=1)

    # Rename target for clarity
    df = df.rename(columns={"diagnosis": "label"})

    # ── Label Encoding ──────────────────────────────────────────
    # Malignant (M) → 1 | Benign (B) → 0
    le = LabelEncoder()
    Y = le.fit_transform(df["label"].values)
    print(f"\n[Preprocess] Encoded classes : {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── Feature Matrix ──────────────────────────────────────────
    # Drop non-predictive columns
    X_raw = df.drop(labels=["label", "id"], axis=1, errors="ignore")
    print(f"[Preprocess] Feature count   : {X_raw.shape[1]}")

    # ── MinMax Scaling ──────────────────────────────────────────
    # X_scaled = (X - X_min) / (X_max - X_min)  →  [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ── Train / Test Split ──────────────────────────────────────
    # Stratified 75 / 25 split to preserve class ratios
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, Y,
        test_size=0.25,
        random_state=42,
        stratify=Y,
    )
    print(f"[Preprocess] Training split  : {x_train.shape}")
    print(f"[Preprocess] Testing split   : {x_test.shape}")
    return x_train, x_test, y_train, y_test


# ────────────────────────────────────────────────────────────────
# 3.  NEURAL NETWORK ARCHITECTURE
# ────────────────────────────────────────────────────────────────

def build_model(input_dim: int) -> tf.keras.Model:
    """
    Construct the MLP architecture:

        Dense(128, ReLU)  ← Input Layer  (30 features)
        Dropout(0.5)      ← Regularization
        Dense(64,  ReLU)  ← Hidden Layer
        Dropout(0.5)      ← Regularization
        Dense(1,   Sigmoid) ← Binary Output  P(Malignant)

    Parameters
    ----------
    input_dim : int
        Number of input features.

    Returns
    -------
    model : compiled tf.keras.Model
    """
    model = Sequential([
        # Layer 1 — Input + Dense representation
        Dense(128, input_dim=input_dim, activation="relu",
              name="input_dense"),

        # Layer 2 — Dropout regularization (50%)
        Dropout(0.5, name="dropout_1"),

        # Layer 3 — Hidden dense layer
        Dense(64, activation="relu", name="hidden_dense"),

        # Layer 4 — Dropout regularization (50%)
        Dropout(0.5, name="dropout_2"),

        # Layer 5 — Binary classification output
        # Sigmoid → P(Malignant) ∈ [0, 1]
        Dense(1, activation="sigmoid", name="output"),
    ], name="BreastCancerMLP")

    model.compile(
        loss="binary_crossentropy",   # H(y, ŷ) = −[y·log(ŷ) + (1−y)·log(1−ŷ)]
        optimizer="adam",             # Adaptive Moment Estimation
        metrics=["accuracy"],
    )

    print("\n[Model] Architecture Summary:")
    model.summary()
    return model


# ────────────────────────────────────────────────────────────────
# 4.  MODEL TRAINING
# ────────────────────────────────────────────────────────────────

def train_model(model, x_train, y_train, x_test, y_test):
    """
    Train the compiled model for 100 epochs with batch size 64.

    Returns
    -------
    history : keras.callbacks.History
    """
    print("\n[Training] Starting training sequence ...")
    history = model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(x_test, y_test),
        verbose=1,
    )
    print("[Training] Complete.")
    return history


# ────────────────────────────────────────────────────────────────
# 5.  METRIC VISUALISATION
# ────────────────────────────────────────────────────────────────

def plot_training_history(history) -> None:
    """Plot and save training / validation loss and accuracy curves."""
    epochs_range = range(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Model Training Performance", fontsize=14, fontweight="bold")

    # Loss plot
    axes[0].plot(epochs_range, history.history["loss"],    "b-", lw=2, label="Training Loss")
    axes[0].plot(epochs_range, history.history["val_loss"], "r-", lw=2, label="Validation Loss")
    axes[0].set_title("Objective Function Convergence (Loss)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs_range, history.history["accuracy"],     "b-", lw=2, label="Training Accuracy")
    axes[1].plot(epochs_range, history.history["val_accuracy"], "r-", lw=2, label="Validation Accuracy")
    axes[1].set_title("Accuracy Trajectory")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("model_training_performance.png", dpi=150)
    plt.close()
    print("[Visualisation] Saved → model_training_performance.png")


# ────────────────────────────────────────────────────────────────
# 6.  INFERENCE & EVALUATION
# ────────────────────────────────────────────────────────────────

def evaluate_model(model, x_test, y_test) -> None:
    """
    Run inference on the test split and print:
      - Classification report (precision, recall, F1)
      - Seaborn confusion matrix heatmap
    """
    # Predict probabilities and binarise at P ≥ 0.5
    y_pred_probs = model.predict(x_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # ── Classification Report ──────────────────────────────────
    print("\n" + "=" * 60)
    print("CLASSIFICATION EVALUATION REPORT")
    print("=" * 60)
    print(classification_report(
        y_test, y_pred,
        target_names=["Benign (0)", "Malignant (1)"],
    ))

    # ── Confusion Matrix ───────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Benign", "Predicted Malignant"],
        yticklabels=["Actual Benign",    "Actual Malignant"],
        annot_kws={"size": 14},
        linewidths=0.5,
    )
    plt.title("Classification Performance Confusion Matrix", fontsize=13)
    plt.ylabel("True Observation Label")
    plt.xlabel("Model Assigned Label")
    plt.tight_layout()
    plt.savefig("evaluation_confusion_matrix.png", dpi=150)
    plt.close()
    print("[Evaluation] Saved → evaluation_confusion_matrix.png")

    # Summary stats
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True  Negatives (Benign correctly identified)    : {tn}")
    print(f"  True  Positives (Malignant correctly identified) : {tp}")
    print(f"  False Positives (Benign mis-classified)          : {fp}")
    print(f"  False Negatives (Malignant mis-classified) ⚠    : {fn}")
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\n  Sensitivity (Recall for Malignant) : {sensitivity:.4f}")
    print(f"  Specificity (Recall for Benign)    : {specificity:.4f}")


# ────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ────────────────────────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  Breast Cancer Classification — Deep Learning        ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # Step 1 — Load & explore
    df = load_and_explore("data.csv")
    run_eda(df)

    # Step 2 — Preprocess
    x_train, x_test, y_train, y_test = preprocess(df)

    # Step 3 — Build model
    model = build_model(input_dim=x_train.shape[1])

    # Step 4 — Train
    history = train_model(model, x_train, y_train, x_test, y_test)

    # Step 5 — Visualise training curves
    plot_training_history(history)

    # Step 6 — Evaluate
    evaluate_model(model, x_test, y_test)

    # Optional — Save trained model
    model.save("breast_cancer_model.h5")
    print("\n[Done] Model saved → breast_cancer_model.h5")
    print("[Done] All performance assets preserved to project directory.\n")


if __name__ == "__main__":
    main()
