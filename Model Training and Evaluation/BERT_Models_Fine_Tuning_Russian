import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    balanced_accuracy_score,
)

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

# Optional: try to use CodeCarbon if installed (otherwise ignore)
try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False


# =========================
# 1) DATA LOADING
# =========================
def load_single_dataset(
    csv_file_path: str,
    text_col: str = "text",
    label_col: str = "label",
    positive_label: str = "stereotype",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Load a single CSV dataset and convert to binary labels:
    1 for `positive_label`, 0 for everything else.
    Split into train / test.
    """

    df = pd.read_csv(csv_file_path)

    # Keep only the columns we actually need
    needed_cols = [text_col, label_col]
    if "group" in df.columns:
        needed_cols.append("group")
    df = df[needed_cols].copy()

    # Map labels to {1, 0}
    label2id = {
        lab: (1 if lab == positive_label else 0)
        for lab in df[label_col].unique()
    }
    df["label"] = df[label_col].map(label2id)

    # Remove original string label column if different from "label"
    if label_col != "label":
        df = df.drop(columns=[label_col])

    # Add dataset name for later tracking (optional)
    df["data_name"] = Path(csv_file_path).stem

    # Train / test split (stratified)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    print("Train size:", len(train_df))
    print("Test size:", len(test_df))
    print("Label distribution (train):")
    print(train_df["label"].value_counts(normalize=True))
    print("Label distribution (test):")
    print(test_df["label"].value_counts(normalize=True))

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    

# =========================
# 2) TRAIN MODEL
# =========================
def train_model(
    train_df: pd.DataFrame,
    model_name: str,
    output_dir: str,
    batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    seed: int = 42,
):
    """
    Fine-tune a HF model on train_df.
    Splits train_df into train/val internally.
    """

    np.random.seed(seed)

    num_labels = train_df["label"].nunique()
    print(f"Number of unique labels: {num_labels}")

    tracker = EmissionsTracker() if HAS_CODECARBON else None
    if tracker is not None:
        tracker.start()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # If the tokenizer has no pad token (rare for ruBERT/bert, but safe to check)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=512,
        )

    # Split into train / validation
    train_split, val_split = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df["label"],
        random_state=seed,
    )

    ds_train = Dataset.from_pandas(train_split).map(
        tokenize_function, batched=True
    )
    ds_train = ds_train.map(lambda e: {"labels": e["label"]})

    ds_val = Dataset.from_pandas(val_split).map(
        tokenize_function, batched=True
    )
    ds_val = ds_val.map(lambda e: {"labels": e["label"]})

    print("Sample tokenized train example:", ds_train[0])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro"
        )
        balanced_acc = balanced_accuracy_score(labels, preds)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": balanced_acc,
        }

    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if tracker is not None:
        emissions = tracker.stop()
        print(f"Estimated total emissions: {emissions} kg CO2")

    print(f"Model saved to: {output_dir}")
    return output_dir


# =========================
# 3) EVALUATE MODEL
# =========================
def evaluate_model(
    test_df: pd.DataFrame,
    model_dir: str,
    result_output_dir: str,
    seed: int = 42,
):
    """
    Evaluate a fine-tuned model on test_df.
    Saves:
    - full_results.csv (per-example)
    - classification_report.csv
    """

    np.random.seed(seed)

    num_labels = test_df["label"].nunique()
    print(f"Number of unique labels in test set: {num_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # CPU: device=-1; GPU: device=0
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        truncation=True,
        max_length=512,
    )

    texts = test_df["text"].tolist()
    predictions_raw = clf(texts, return_all_scores=True)

    pred_labels = []
    pred_probs = []

    for scores in predictions_raw:
        best = max(scores, key=lambda x: x["score"])
        # label is like 'LABEL_0', 'LABEL_1', etc.
        label_id = int(best["label"].split("_")[-1])
        pred_labels.append(label_id)
        pred_probs.append(best["score"])

    y_true = test_df["label"].tolist()

    # Build results dataframe
    results_df = pd.DataFrame(
        {
            "text": test_df["text"],
            "predicted_label": pred_labels,
            "predicted_probability": pred_probs,
            "actual_label": y_true,
        }
    )

    if "group" in test_df.columns:
        results_df["group"] = test_df["group"]

    if "data_name" in test_df.columns:
        results_df["dataset_name"] = test_df["data_name"]
    else:
        results_df["dataset_name"] = Path(model_dir).name

    os.makedirs(result_output_dir, exist_ok=True)

    full_results_path = os.path.join(result_output_dir, "full_results.csv")
    results_df.to_csv(full_results_path, index=False)
    print(f"Saved full results to {full_results_path}")

    report_dict = classification_report(y_true, pred_labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    report_path = os.path.join(result_output_dir, "classification_report.csv")
    report_df.to_csv(report_path)
    print(f"Saved classification report to {report_path}")

    return report_df

if __name__ == "__main__":
    train_df, test_df = load_single_dataset(
        csv_file_path="COMP0173_Data/rubist.csv",
        text_col="text",
        label_col="category",
        positive_label="stereotype",
        test_size=0.2,
        random_state=42,
    )

    model_dir = train_model(
        train_df=train_df,
        model_name="DeepPavlov/rubert-base-cased",
        output_dir="COMP0173_Results/model_output_rubert_rubist",
        batch_size=16,
        num_epochs=3,
        learning_rate=2e-5,
        seed=42,
    )

    evaluate_model(
        test_df=test_df,
        model_dir=model_dir,
        result_output_dir="COMP0173_Results/result_output_rubert_rubist",
        seed=42,
    )