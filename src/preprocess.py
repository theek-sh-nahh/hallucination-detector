import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import json


# ── Label map ────────────────────────────────────────────────────
# 0 = factual          (TruthfulQA correct answers)
# 1 = hallucinated     (TruthfulQA incorrect answers / AI fabrications)
# 2 = partially_true   (mixed or hedged answers)
# 3 = overconfident    (confidently wrong, no hedging language)

LABEL_MAP = {
    "factual": 0,
    "hallucinated": 1,
    "partially_true": 2,
    "overconfident": 3
}

HEDGE_WORDS = [
    "i think", "i believe", "probably", "possibly", "might",
    "could be", "i'm not sure", "i am not certain", "perhaps",
    "it seems", "i'm not 100%"
]


def load_truthfulqa():
    """
    Load TruthfulQA from HuggingFace datasets.
    Returns a list of dicts with 'text' and 'label' keys.
    """
    print("Loading TruthfulQA dataset...")
    # dataset = load_dataset("truthful_qa", "generation", trust_remote_code=True)
    dataset = load_dataset("truthfulqa/truthful_qa", "generation")
    
    samples = []
    split_name = "validation" if "validation" in dataset else list(dataset.keys())[0]
    for item in dataset[split_name]:
    # for item in dataset["validation"]:
        question = item["question"]
        
        # Correct answers → factual (label 0)
        for ans in item["correct_answers"]:
            if ans.strip():
                samples.append({
                    "text": f"Q: {question} A: {ans}",
                    "label": "factual"
                })
        
        # Incorrect answers → hallucinated (label 1)
        for ans in item["incorrect_answers"]:
            if ans.strip():
                text = f"Q: {question} A: {ans}"
                label = classify_incorrect(ans)
                samples.append({
                    "text": text,
                    "label": label
                })
    
    print(f"  Loaded {len(samples)} samples from TruthfulQA")
    return samples


def classify_incorrect(answer_text):
    """
    Sub-classify incorrect answers into hallucinated,
    partially_true, or overconfident using heuristics.
    """
    text_lower = answer_text.lower()
    
    has_hedge = any(word in text_lower for word in HEDGE_WORDS)
    is_short   = len(answer_text.split()) < 8
    
    if has_hedge:
        return "partially_true"     # hedging = aware of uncertainty
    elif is_short:
        return "overconfident"      # blunt wrong answer = overconfident
    else:
        return "hallucinated"       # confident, detailed, wrong = hallucination


def load_custom_samples(filepath):
    """
    Load self-generated or API-collected samples from a JSON file.
    Expected format:
    [{"text": "Q: ... A: ...", "label": "hallucinated"}, ...]
    """
    if not os.path.exists(filepath):
        print(f"  Custom samples file not found: {filepath} — skipping")
        return []
    
    with open(filepath, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    print(f"  Loaded {len(samples)} custom samples from {filepath}")
    return samples


def build_dataframe(samples):
    """
    Convert list of dicts to a clean DataFrame.
    Encodes labels as integers.
    """
    df = pd.DataFrame(samples)
    df = df.drop_duplicates(subset=["text"])
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin(LABEL_MAP.keys())]
    
    df["label_id"] = df["label"].map(LABEL_MAP)
    df = df.reset_index(drop=True)
    
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Class distribution:\n{df['label'].value_counts()}")
    return df


def generate_embeddings(texts, model_name="all-MiniLM-L6-v2", batch_size=64):
    """
    Generate Sentence-BERT embeddings for a list of texts.
    Returns numpy array of shape (n_samples, 384).
    """
    print(f"  Loading SentenceTransformer: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"  Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


def split_data(embeddings, labels, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split into train / val / test sets.
    Stratified to preserve class balance.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        embeddings, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    adjusted_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=adjusted_val,
        stratify=y_temp,
        random_state=random_state
    )
    
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed_data(output_dir, X_train, X_val, X_test,
                         y_train, y_val, y_test, df):
    """
    Save all splits as .npy files and the full DataFrame as CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(output_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"),   y_val)
    np.save(os.path.join(output_dir, "y_test.npy"),  y_test)
    
    df.to_csv(os.path.join(output_dir, "dataset.csv"), index=False)
    
    print(f"  All splits saved to: {output_dir}")


def run_pipeline(raw_data_dir="data/raw", output_dir="data/processed"):
    """
    Full preprocessing pipeline — call this from the Colab notebook.
    """
    print("=" * 50)
    print("HALLUCINATION DETECTOR — PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # 1. Load data
    samples = load_truthfulqa()
    custom  = load_custom_samples(os.path.join(raw_data_dir, "custom_samples.json"))
    all_samples = samples + custom
    
    # 2. Build DataFrame
    print("\n[2] Building DataFrame...")
    df = build_dataframe(all_samples)
    
    # 3. Generate embeddings
    print("\n[3] Generating embeddings...")
    embeddings = generate_embeddings(df["text"].tolist())
    labels     = df["label_id"].values
    
    # 4. Split
    print("\n[4] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(embeddings, labels)
    
    # 5. Save
    print("\n[5] Saving processed data...")
    save_processed_data(output_dir, X_train, X_val, X_test,
                        y_train, y_val, y_test, df)
    
    print("\nPipeline complete.")
    return X_train, X_val, X_test, y_train, y_val, y_test, df


if __name__ == "__main__":
    run_pipeline()