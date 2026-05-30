import numpy as np
import json
import os


LABEL_NAMES = {
    0: "factual",
    1: "hallucinated",
    2: "partially_true",
    3: "overconfident"
}


def normalize_errors(errors, min_e=None, max_e=None):
    """
    Normalize reconstruction errors to [0, 1].
    min/max can be provided (from training set) for consistent scaling.
    """
    if min_e is None:
        min_e = errors.min()
    if max_e is None:
        max_e = errors.max()
    return (errors - min_e) / (max_e - min_e + 1e-8), min_e, max_e


def fuse_scores(lstm_probs, ae_errors,
                ae_min=None, ae_max=None,
                lstm_weight=0.75, ae_weight=0.25):
    """
    Combine BiLSTM class probabilities with AE anomaly scores.

    lstm_probs : (N, 4) softmax probabilities from BiLSTM
    ae_errors  : (N,)   reconstruction errors from Autoencoder

    Strategy:
    - LSTM gives us a 4-class probability distribution
    - AE gives us an anomaly score (high = hallucinated)
    - We boost the hallucination probability using the AE score
    - Final confidence = weighted combination

    Returns:
    - final_preds   : (N,) predicted class indices
    - halluc_conf   : (N,) hallucination confidence 0-100%
    - fused_probs   : (N, 4) adjusted probability distribution
    """
    ae_scores_norm, ae_min, ae_max = normalize_errors(
        ae_errors, ae_min, ae_max
    )

    # Clone LSTM probs and boost hallucination channel with AE score
    fused = lstm_probs.copy()
    fused[:, 1] = (
        lstm_weight * lstm_probs[:, 1] +
        ae_weight   * ae_scores_norm
    )

    # Renormalize so probabilities sum to 1
    fused = fused / fused.sum(axis=1, keepdims=True)

    final_preds = fused.argmax(axis=1)

    # Hallucination confidence = P(hallucinated) + 0.5*P(overconfident)
    # Overconfident answers are a softer form of hallucination
    halluc_conf = (fused[:, 1] + 0.5 * fused[:, 3]) * 100

    return final_preds, halluc_conf, fused, ae_min, ae_max


def interpret_result(pred_class, halluc_confidence):
    """
    Human-readable interpretation of a single prediction.
    """
    label = LABEL_NAMES.get(pred_class, "unknown")

    if halluc_confidence < 25:
        risk_level = "LOW"
        color_hint = "green"
    elif halluc_confidence < 50:
        risk_level = "MODERATE"
        color_hint = "yellow"
    elif halluc_confidence < 75:
        risk_level = "HIGH"
        color_hint = "orange"
    else:
        risk_level = "CRITICAL"
        color_hint = "red"

    return {
        "predicted_class":      label,
        "hallucination_confidence": round(float(halluc_confidence), 1),
        "risk_level":           risk_level,
        "color":                color_hint,
        "interpretation": (
            f"This answer is classified as '{label}' with "
            f"{halluc_confidence:.1f}% hallucination confidence "
            f"({risk_level} risk)."
        )
    }


def save_fusion_params(ae_min, ae_max, lstm_weight, ae_weight, save_dir):
    """Save fusion parameters so the app can load them at inference."""
    params = {
        "ae_min":       float(ae_min),
        "ae_max":       float(ae_max),
        "lstm_weight":  lstm_weight,
        "ae_weight":    ae_weight
    }
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "fusion_params.json")
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Fusion params saved to {path}")
    return path


def load_fusion_params(save_dir):
    """Load fusion parameters for inference."""
    path = os.path.join(save_dir, "fusion_params.json")
    with open(path, "r") as f:
        params = json.load(f)
    return params


if __name__ == "__main__":
    print("fusion.py loaded — import and use via Colab notebook.")