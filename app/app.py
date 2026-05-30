import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
import gradio as gr
from sentence_transformers import SentenceTransformer

from src.lstm_model import BiLSTMClassifier
from src.autoencoder import HallucinationAutoencoder, eval_ae
from src.fusion import fuse_scores, interpret_result

# ── Config ────────────────────────────────────────────────────────

MODELS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models')
DEVICE      = torch.device('cpu')   # app runs on CPU locally
LABEL_NAMES = ["factual", "hallucinated", "partially_true", "overconfident"]

LABEL_COLORS = {
    "factual":          "#22c55e",   # green
    "hallucinated":     "#ef4444",   # red
    "partially_true":   "#f97316",   # orange
    "overconfident":    "#a855f7"    # purple
}

LABEL_DESCRIPTIONS = {
    "factual":
        "This answer appears to be accurate and well-grounded.",
    "hallucinated":
        "This answer contains fabricated or false information.",
    "partially_true":
        "This answer mixes correct and incorrect information.",
    "overconfident":
        "This answer states uncertain claims with unwarranted confidence."
}


# ── Model Loading ─────────────────────────────────────────────────

def load_models():
    """Load all models and params at startup."""
    print("Loading models...")

    # Sentence-BERT embedder
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # BiLSTM
    lstm = BiLSTMClassifier(
        input_dim=384, hidden_dim=128,
        num_layers=2, num_classes=4, dropout=0.4
    )
    lstm_path = os.path.join(MODELS_DIR, 'lstm', 'lstm_final.pt')
    lstm.load_state_dict(
        torch.load(lstm_path, map_location=DEVICE)
    )
    lstm.eval()
    print(f"  LSTM loaded")

    # Autoencoder
    ae = HallucinationAutoencoder(
        input_dim=384, latent_dim=16, dropout=0.2
    )
    ae_path = os.path.join(MODELS_DIR, 'autoencoder', 'ae_final.pt')
    ae.load_state_dict(
        torch.load(ae_path, map_location=DEVICE)
    )
    ae.eval()
    print(f"  Autoencoder loaded")

    # Fusion params
    with open(os.path.join(MODELS_DIR, 'fusion_params.json')) as f:
        fusion_params = json.load(f)

    # AE threshold
    with open(os.path.join(MODELS_DIR, 'ae_threshold.json')) as f:
        threshold_data = json.load(f)
    threshold = threshold_data['threshold']

    print("All models ready.")
    return embedder, lstm, ae, fusion_params, threshold


# Load once at startup
embedder, lstm_model, ae_model, fusion_params, ae_threshold = load_models()


# ── Inference ─────────────────────────────────────────────────────

def detect_hallucination(question: str, answer: str):
    """
    Main inference function called by Gradio.
    Takes a question + answer, returns classification and confidence.
    """
    if not question.strip() or not answer.strip():
        return (
            "⚠️ Please enter both a question and an answer.",
            "", "", "", ""
        )

    # Format input the same way as training
    text = f"Q: {question.strip()} A: {answer.strip()}"

    # Generate embedding
    embedding = embedder.encode(
        [text], convert_to_numpy=True
    )   # shape (1, 384)

    # BiLSTM prediction
    with torch.no_grad():
        x      = torch.FloatTensor(embedding).unsqueeze(1)
        logits = lstm_model(x)
        probs  = torch.softmax(logits, dim=1).numpy()

    # AE reconstruction error
    ae_error = eval_ae(ae_model, embedding, DEVICE)

    # Fuse scores
    final_preds, halluc_conf, fused_probs, _, _ = fuse_scores(
        probs,
        ae_error,
        ae_min=fusion_params['ae_min'],
        ae_max=fusion_params['ae_max'],
        lstm_weight=fusion_params['lstm_weight'],
        ae_weight=fusion_params['ae_weight']
    )

    result = interpret_result(int(final_preds[0]), halluc_conf[0])

    # Build outputs
    label       = result['predicted_class']
    confidence  = result['hallucination_confidence']
    risk        = result['risk_level']
    color       = LABEL_COLORS.get(label, '#6b7280')
    description = LABEL_DESCRIPTIONS.get(label, '')

    # Confidence meter bar (HTML)
    meter_html = build_meter_html(confidence, label, color)

    # Breakdown text
    breakdown = build_breakdown(probs[0], fused_probs[0],
                                float(ae_error[0]), confidence)

    return (
        meter_html,
        f"{label.upper().replace('_', ' ')}",
        f"{risk} RISK",
        description,
        breakdown
    )


def build_meter_html(confidence, label, color):
    """Build an HTML confidence meter bar."""
    bar_width = min(max(confidence, 2), 100)

    # Color gradient based on confidence
    if confidence < 25:
        bar_color = "#22c55e"
    elif confidence < 50:
        bar_color = "#eab308"
    elif confidence < 75:
        bar_color = "#f97316"
    else:
        bar_color = "#ef4444"

    html = f"""
    <div style="font-family: sans-serif; padding: 16px;">
        <div style="margin-bottom: 8px; font-size: 14px;
                    color: #6b7280;">
            Hallucination Confidence
        </div>
        <div style="background: #e5e7eb; border-radius: 9999px;
                    height: 28px; width: 100%; overflow: hidden;">
            <div style="
                width: {bar_width}%;
                height: 100%;
                background: {bar_color};
                border-radius: 9999px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 14px;
                transition: width 0.5s ease;
            ">
                {confidence:.1f}%
            </div>
        </div>
        <div style="display: flex; justify-content: space-between;
                    margin-top: 4px; font-size: 12px; color: #9ca3af;">
            <span>0% (Factual)</span>
            <span>100% (Hallucinated)</span>
        </div>
    </div>
    """
    return html


def build_breakdown(lstm_probs, fused_probs, ae_error, halluc_conf):
    """Build a detailed breakdown string."""
    lines = [
        "── Model Breakdown ──────────────────────",
        "",
        "BiLSTM probabilities:",
    ]
    for i, name in enumerate(LABEL_NAMES):
        bar = "█" * int(lstm_probs[i] * 20)
        lines.append(f"  {name:<18} {lstm_probs[i]:.3f}  {bar}")

    lines += [
        "",
        "Fused probabilities:",
    ]
    for i, name in enumerate(LABEL_NAMES):
        bar = "█" * int(fused_probs[i] * 20)
        lines.append(f"  {name:<18} {fused_probs[i]:.3f}  {bar}")

    lines += [
        "",
        f"AE reconstruction error: {ae_error:.6f}",
        f"Final hallucination score: {halluc_conf:.1f}%",
    ]
    return "\n".join(lines)


# ── Gradio UI ─────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="AI Hallucination Detector",
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown("""
        # 🔍 AI Hallucination Detector
        **Hybrid BiLSTM + Autoencoder** — detects hallucinated,
        overconfident, partially true, and factual AI answers.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="e.g. Who invented the telephone?",
                    lines=3
                )
                answer_input = gr.Textbox(
                    label="AI-generated Answer",
                    placeholder="Paste the AI answer here...",
                    lines=5
                )
                submit_btn = gr.Button(
                    "Analyse", variant="primary", size="lg"
                )

                gr.Examples(
                    examples=[
                        [
                            "Who invented the telephone?",
                            "Alexander Graham Bell is widely credited with inventing the telephone in 1876."
                        ],
                        [
                            "What is the boiling point of water?",
                            "Water boils at exactly 90 degrees Celsius at sea level."
                        ],
                        [
                            "Who wrote Romeo and Juliet?",
                            "I believe it was Shakespeare, though I'm not entirely certain of the exact date."
                        ],
                        [
                            "What is the capital of Australia?",
                            "The capital of Australia is Sydney, which is also the largest city."
                        ],
                    ],
                    inputs=[question_input, answer_input],
                    label="Try these examples"
                )

            with gr.Column(scale=1):
                meter_output = gr.HTML(label="Confidence Meter")

                with gr.Row():
                    label_output = gr.Textbox(
                        label="Classification", interactive=False
                    )
                    risk_output = gr.Textbox(
                        label="Risk Level", interactive=False
                    )

                desc_output = gr.Textbox(
                    label="Interpretation",
                    interactive=False,
                    lines=2
                )
                breakdown_output = gr.Textbox(
                    label="Detailed Breakdown",
                    interactive=False,
                    lines=14
                )

        submit_btn.click(
            fn=detect_hallucination,
            inputs=[question_input, answer_input],
            outputs=[
                meter_output,
                label_output,
                risk_output,
                desc_output,
                breakdown_output
            ]
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False)