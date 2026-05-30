import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)

from src.lstm_model import BiLSTMClassifier, predict as lstm_predict
from src.autoencoder import HallucinationAutoencoder, eval_ae
from src.fusion import fuse_scores, interpret_result

# LABEL_NAMES = ["factual", "hallucinated", "partially_true", "overconfident"]
LABEL_NAMES = ["factual", "hallucinated", "overconfident"]


# ── Load Models ───────────────────────────────────────────────────

def load_lstm(model_path, device, hidden_dim=128, dropout=0.4):
    model = BiLSTMClassifier(
        input_dim=384,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_classes=4,
        dropout=dropout
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  LSTM loaded from {model_path}")
    return model


def load_autoencoder(model_path, device, latent_dim=16, dropout=0.2):
    model = HallucinationAutoencoder(
        input_dim=384,
        latent_dim=latent_dim,
        dropout=dropout
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  AE loaded from {model_path}")
    return model


def load_fusion_params(params_path):
    with open(params_path, "r") as f:
        params = json.load(f)
    print(f"  Fusion params: {params}")
    return params


# ── Evaluation ────────────────────────────────────────────────────

def evaluate_lstm_alone(lstm_model, X_test, y_test, device):
    """Evaluate BiLSTM classifier in isolation."""
    print("\n[LSTM STANDALONE EVALUATION]")

    preds, probs = lstm_predict(lstm_model, X_test, device)

    print(classification_report(
        y_test, preds,
        target_names=LABEL_NAMES,
        zero_division=0
    ))

    macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    print(f"  Macro F1: {macro_f1:.4f}")

    return preds, probs, macro_f1


def evaluate_ae_alone(ae_model, X_test, y_test, device, threshold):
    """Evaluate AE binary classifier (factual vs hallucinated) in isolation."""
    print("\n[AUTOENCODER STANDALONE EVALUATION]")

    errors      = eval_ae(ae_model, X_test, device)
    ae_preds    = (errors > threshold).astype(int)
    binary_true = (y_test == 1).astype(int)

    print(classification_report(
        binary_true, ae_preds,
        target_names=["not_hallucinated", "hallucinated"],
        zero_division=0
    ))

    try:
        auc = roc_auc_score(binary_true, errors)
        print(f"  ROC-AUC: {auc:.4f}")
    except Exception:
        auc = None

    return errors, ae_preds, auc


def evaluate_fused(lstm_probs, ae_errors, y_test, params):
    """Evaluate the fused system."""
    print("\n[FUSED SYSTEM EVALUATION]")

    final_preds, halluc_conf, fused_probs, _, _ = fuse_scores(
        lstm_probs,
        ae_errors,
        ae_min=params["ae_min"],
        ae_max=params["ae_max"],
        lstm_weight=params["lstm_weight"],
        ae_weight=params["ae_weight"]
    )

    print(classification_report(
        y_test, final_preds,
        target_names=LABEL_NAMES,
        zero_division=0
    ))

    macro_f1 = f1_score(
        y_test, final_preds, average='macro', zero_division=0
    )
    print(f"  Macro F1:              {macro_f1:.4f}")
    print(f"  Mean halluc confidence: {halluc_conf.mean():.1f}%")

    return final_preds, halluc_conf, fused_probs, macro_f1


# ── Plots ─────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cmap='Blues'
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved: {save_path}")


def plot_confidence_distribution(halluc_conf, y_test, save_path):
    """
    Show how hallucination confidence distributes
    across actual classes — key diagnostic for the meter.
    """
    plt.figure(figsize=(10, 5))
    colors = ['green', 'red', 'orange', 'purple']

    for cls_idx, (cls_name, color) in enumerate(
        zip(LABEL_NAMES, colors)
    ):
        mask = y_test == cls_idx
        if mask.sum() > 0:
            plt.hist(
                halluc_conf[mask], bins=30,
                alpha=0.5, label=cls_name, color=color
            )

    plt.xlabel('Hallucination Confidence (%)')
    plt.ylabel('Count')
    plt.title('Hallucination Confidence Distribution by True Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved: {save_path}")


# ── Full Pipeline ─────────────────────────────────────────────────

def run_full_evaluation(
    lstm_path, ae_path, params_path, threshold_path,
    X_test, y_test, device, output_dir
):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 55)
    print("HALLUCINATION DETECTOR — FULL EVALUATION")
    print("=" * 55)

    # Load models
    print("\n[Loading models...]")
    lstm_model = load_lstm(lstm_path, device)
    ae_model   = load_autoencoder(ae_path, device)
    params     = load_fusion_params(params_path)

    with open(threshold_path) as f:
        threshold_data = json.load(f)
    threshold = threshold_data["threshold"]
    print(f"  AE threshold: {threshold:.6f}")

    # Run evaluations
    lstm_preds, lstm_probs, lstm_f1 = evaluate_lstm_alone(
        lstm_model, X_test, y_test, device
    )

    ae_errors, ae_preds, ae_auc = evaluate_ae_alone(
        ae_model, X_test, y_test, device, threshold
    )

    final_preds, halluc_conf, fused_probs, fused_f1 = evaluate_fused(
        lstm_probs, ae_errors, y_test, params
    )

    # Plots
    print("\n[Generating plots...]")
    plot_confusion_matrix(
        y_test, lstm_preds,
        "LSTM Confusion Matrix",
        f"{output_dir}/lstm_confusion.png"
    )
    plot_confusion_matrix(
        y_test, final_preds,
        "Fused System Confusion Matrix",
        f"{output_dir}/fused_confusion.png"
    )
    plot_confidence_distribution(
        halluc_conf, y_test,
        f"{output_dir}/confidence_distribution.png"
    )

    # Save summary
    summary = {
        "lstm_macro_f1":    round(lstm_f1, 4),
        "ae_roc_auc":       round(ae_auc, 4) if ae_auc else None,
        "fused_macro_f1":   round(fused_f1, 4),
        "mean_halluc_conf": round(float(halluc_conf.mean()), 2),
        "threshold":        round(threshold, 6)
    }

    summary_path = f"{output_dir}/evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*55}")
    print("EVALUATION SUMMARY")
    print(f"{'='*55}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary, halluc_conf


if __name__ == "__main__":
    print("evaluate.py — import and run via Colab notebook.")