# BERT_Models_Fine_Tuning.py

from pathlib import Path
import os
import logging
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
import torch


# Paths and logging

# This file is in: .../HEARTS-Text-Stereotype-Detection/Model Training and Evaluation
BASE_DIR = Path(__file__).resolve().parent          # folder with the CSVs
DATA_DIR = BASE_DIR

os.environ["HUGGINGFACE_TRAINER_ENABLE_PROGRESS_BAR"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)


# Data helpers
def data_loader(csv_file_path, labelling_criteria, dataset_name, sample_size, num_examples):
    
    """
    Load a CSV, binarise the label column wrt `labelling_criteria`,
    stratified sample to `sample_size` (if smaller than dataset),
    and return a stratified 80/20 train/test split.
    """

    csv_path = DATA_DIR / csv_file_path
    print(f"Loading: {csv_path}")

    combined_data = pd.read_csv(csv_path, usecols=["text", "label", "group"])

    # Binarise labels: 1 if == labelling_criteria else 0
    label2id = {
        label: (1 if label == labelling_criteria else 0)
        for label in combined_data["label"].unique()
    }
    combined_data["label"] = combined_data["label"].map(label2id)

    combined_data["data_name"] = dataset_name

    # Optional downsampling
    if sample_size >= len(combined_data):
        sampled_data = combined_data
    else:
        sample_proportion = sample_size / len(combined_data)
        sampled_data, _ = train_test_split(
            combined_data,
            train_size=sample_proportion,
            stratify=combined_data["label"],
            random_state=42,
        )

    # Train/test split
    train_data, test_data = train_test_split(
        sampled_data,
        test_size=0.2,
        random_state=42,
        stratify=sampled_data["label"],
    )

    print("First few examples from the training data:")
    print(train_data.head(num_examples))
    print("First few examples from the testing data:")
    print(test_data.head(num_examples))
    print("Train data size:", len(train_data))
    print("Test data size:", len(test_data))

    return train_data, test_data


def merge_datasets(
    train_data_candidate,
    test_data_candidate,
    train_data_established,
    test_data_established,
    num_examples,
):
    """
    Merge a 'candidate' dataset with an 'established' one
    (train + train, test + test).
    """
    merged_train_data = pd.concat(
        [train_data_candidate, train_data_established], ignore_index=True
    )
    merged_test_data = pd.concat(
        [test_data_candidate, test_data_established], ignore_index=True
    )

    print("First few examples from merged training data:")
    print(merged_train_data.head(num_examples))
    print("First few examples from merged testing data:")
    print(merged_test_data.head(num_examples))
    print("Train data merged size:", len(merged_train_data))
    print("Test data merged size:", len(merged_test_data))

    return merged_train_data, merged_test_data


# Model training
def train_model(
    train_data,
    model_path,
    batch_size,
    epoch,
    learning_rate,
    model_output_base_dir,
    dataset_name,
    seed,
):
    """
    Fine-tune a HF model (e.g. ALBERT, DistilBERT, BERT) on `train_data`.
    """
    np.random.seed(seed)

    num_labels = len(train_data["label"].unique())
    print(f"Number of unique labels: {num_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # GPT-style models need a pad token (not used here, but kept for completeness)
    if model_path.startswith("gpt"):
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=512,
        )

    # Train/validation split
    train_split, val_split = train_test_split(
        train_data, test_size=0.2, random_state=42
    )

    tokenized_train = (
        Dataset.from_pandas(train_split)
        .map(tokenize_function, batched=True)
        .map(lambda examples: {"labels": examples["label"]})
    )
    print("Sample tokenized input from train:", tokenized_train[0])

    tokenized_val = (
        Dataset.from_pandas(val_split)
        .map(tokenize_function, batched=True)
        .map(lambda examples: {"labels": examples["label"]})
    )
    print("Sample tokenized input from validation:", tokenized_val[0])

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro"
        )
        balanced_acc = balanced_accuracy_score(labels, predictions)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": balanced_acc,
        }

    model_output_dir = Path(model_output_base_dir) / dataset_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        num_train_epochs=epoch,
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(model_output_dir))

    print(f"Saved fine-tuned model to: {model_output_dir}")

    return str(model_output_dir)


# Evaluation
def evaluate_model(
    test_data,
    model_output_dir,
    result_output_base_dir,
    dataset_name,
    seed,
):
    """
    Evaluate a fine-tuned model on `test_data`, save full predictions and
    classification report as CSV.
    """
    np.random.seed(seed)

    num_labels = len(test_data["label"].unique())
    print(f"Number of unique labels: {num_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_output_dir,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_output_dir)

    if model_output_dir.startswith("gpt"):
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=512,
        )

    tokenized_test = (
        Dataset.from_pandas(test_data)
        .map(tokenize_function, batched=True)
        .map(lambda examples: {"labels": examples["label"]})
    )
    print("Sample tokenized input from test:", tokenized_test[0])

    result_output_dir = Path(result_output_base_dir) / dataset_name
    result_output_dir.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    predictions = pipe(
        test_data["text"].to_list(),
        return_all_scores=True,
    )

    pred_labels = [
        int(max(pred, key=lambda x: x["score"])["label"].split("_")[-1])
        for pred in predictions
    ]
    pred_probs = [
        max(pred, key=lambda x: x["score"])["score"]
        for pred in predictions
    ]
    y_true = test_data["label"].tolist()

    results_df = pd.DataFrame(
        {
            "text": test_data["text"],
            "predicted_label": pred_labels,
            "predicted_probability": pred_probs,
            "actual_label": y_true,
            "group": test_data["group"],
            "dataset_name": test_data["data_name"],
        }
    )

    results_file_path = result_output_dir / "full_results.csv"
    results_df.to_csv(results_file_path, index=False)
    print(f"Saved full results to: {results_file_path}")

    report = classification_report(y_true, pred_labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    report_file_path = result_output_dir / "classification_report.csv"
    df_report.to_csv(report_file_path)
    print(f"Saved classification report to: {report_file_path}")

    return df_report

# Full pipeline: replicate HEARTS 
def run_full_hearts_pipeline():

    # Load all base datasets

    # MGSD (established)
    train_data_mgsd, test_data_mgsd = data_loader(
        csv_file_path="MGSD.csv",
        labelling_criteria="stereotype",
        dataset_name="MGSD",
        sample_size=1_000_000,
        num_examples=5,
    )

    # WinoQueer GPT augmentation (candidate)
    train_data_winoqueer_gpt_augmentation, test_data_winoqueer_gpt_augmentation = data_loader(
        csv_file_path="Winoqueer - GPT Augmentation.csv",
        labelling_criteria="stereotype",
        dataset_name="Winoqueer - GPT Augmentation",
        sample_size=1_000_000,
        num_examples=5,
    )

    # SeeGULL GPT augmentation (candidate)
    train_data_seegull_gpt_augmentation, test_data_seegull_gpt_augmentation = data_loader(
        csv_file_path="SeeGULL - GPT Augmentation.csv",
        labelling_criteria="stereotype",
        dataset_name="SeeGULL - GPT Augmentation",
        sample_size=1_000_000,
        num_examples=5,
    )

    # Build merged datasets

    # MGSD + WinoQueer
    train_data_merged_winoqueer_gpt_augmentation, test_data_merged_winoqueer_gpt_augmentation = merge_datasets(
        train_data_candidate=train_data_winoqueer_gpt_augmentation,
        test_data_candidate=test_data_winoqueer_gpt_augmentation,
        train_data_established=train_data_mgsd,
        test_data_established=test_data_mgsd,
        num_examples=5,
    )

    # MGSD + SeeGULL
    train_data_merged_seegull_gpt_augmentation, test_data_merged_seegull_gpt_augmentation = merge_datasets(
        train_data_candidate=train_data_seegull_gpt_augmentation,
        test_data_candidate=test_data_seegull_gpt_augmentation,
        train_data_established=train_data_mgsd,
        test_data_established=test_data_mgsd,
        num_examples=5,
    )

    # MGSD + WinoQueer + SeeGULL
    train_data_merged_winoqueer_seegull_gpt_augmentation, test_data_merged_winoqueer_seegull_gpt_augmentation = merge_datasets(
        train_data_candidate=train_data_seegull_gpt_augmentation,
        test_data_candidate=test_data_seegull_gpt_augmentation,
        train_data_established=train_data_merged_winoqueer_gpt_augmentation,
        test_data_established=test_data_merged_winoqueer_gpt_augmentation,
        num_examples=5,
    )

    # ALBERT 

    albert_model_base = "model_output_albertv2"
    albert_results_base = "result_output_albertv2"

    # (a) Train on MGSD 
    mgsd_trained_albert = train_model(
        train_data=train_data_mgsd,
        model_path="albert/albert-base-v2",
        batch_size=64,
        epoch=6,
        learning_rate=2e-5,
        model_output_base_dir=albert_model_base,
        dataset_name="mgsd_trained",
        seed=42,
    )
    evaluate_model(test_data_winoqueer_gpt_augmentation, mgsd_trained_albert, albert_results_base + "/mgsd_trained", "winoqueer_gpt_augmentation", 42)
    evaluate_model(test_data_seegull_gpt_augmentation, mgsd_trained_albert, albert_results_base + "/mgsd_trained", "seegull_gpt_augmentation", 42)
    evaluate_model(test_data_mgsd, mgsd_trained_albert, albert_results_base + "/mgsd_trained", "mgsd", 42)
    evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, mgsd_trained_albert, albert_results_base + "/mgsd_trained", "merged_winoqueer_seegull_gpt_augmentation", 42)

    # (b) Train on WinoQueer 
    wino_trained_albert = train_model(
        train_data=train_data_winoqueer_gpt_augmentation,
        model_path="albert/albert-base-v2",
        batch_size=64,
        epoch=6,
        learning_rate=2e-5,
        model_output_base_dir=albert_model_base,
        dataset_name="winoqueer_gpt_augmentation_trained",
        seed=42,
    )
    evaluate_model(test_data_winoqueer_gpt_augmentation, wino_trained_albert, albert_results_base + "/winoqueer_gpt_augmentation_trained", "winoqueer_gpt_augmentation", 42)
    evaluate_model(test_data_seegull_gpt_augmentation, wino_trained_albert, albert_results_base + "/winoqueer_gpt_augmentation_trained", "seegull_gpt_augmentation", 42)
    evaluate_model(test_data_mgsd, wino_trained_albert, albert_results_base + "/winoqueer_gpt_augmentation_trained", "mgsd", 42)
    evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, wino_trained_albert, albert_results_base + "/winoqueer_gpt_augmentation_trained", "merged_winoqueer_seegull_gpt_augmentation", 42)

    # (c) Train on SeeGULL 
    seegull_trained_albert = train_model(
        train_data=train_data_seegull_gpt_augmentation,
        model_path="albert/albert-base-v2",
        batch_size=64,
        epoch=6,
        learning_rate=2e-5,
        model_output_base_dir=albert_model_base,
        dataset_name="seegull_gpt_augmentation_trained",
        seed=42,
    )
    evaluate_model(test_data_winoqueer_gpt_augmentation, seegull_trained_albert, albert_results_base + "/seegull_gpt_augmentation_trained", "winoqueer_gpt_augmentation", 42)
    evaluate_model(test_data_seegull_gpt_augmentation, seegull_trained_albert, albert_results_base + "/seegull_gpt_augmentation_trained", "seegull_gpt_augmentation", 42)
    evaluate_model(test_data_mgsd, seegull_trained_albert, albert_results_base + "/seegull_gpt_augmentation_trained", "mgsd", 42)
    evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, seegull_trained_albert, albert_results_base + "/seegull_gpt_augmentation_trained", "merged_winoqueer_seegull_gpt_augmentation", 42)

    # (d) Train on merged WinoQueer + SeeGULL + MGSD
    merged_trained_albert = train_model(
        train_data=train_data_merged_winoqueer_seegull_gpt_augmentation,
        model_path="albert/albert-base-v2",
        batch_size=64,
        epoch=6,
        learning_rate=2e-5,
        model_output_base_dir=albert_model_base,
        dataset_name="merged_winoqueer_seegull_gpt_augmentation_trained",
        seed=42,
    )
    evaluate_model(test_data_winoqueer_gpt_augmentation, merged_trained_albert, albert_results_base + "/merged_winoqueer_seegull_gpt_augmentation_trained", "winoqueer_gpt_augmentation", 42)
    evaluate_model(test_data_seegull_gpt_augmentation, merged_trained_albert, albert_results_base + "/merged_winoqueer_seegull_gpt_augmentation_trained", "seegull_gpt_augmentation", 42)
    evaluate_model(test_data_mgsd, merged_trained_albert, albert_results_base + "/merged_winoqueer_seegull_gpt_augmentation_trained", "mgsd", 42)
    evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, merged_trained_albert, albert_results_base + "/merged_winoqueer_seegull_gpt_augmentation_trained", "merged_winoqueer_seegull_gpt_augmentation", 42)

    # DistilBERT 

    distil_model_base = "model_output_distilbert"
    distil_results_base = "result_output_distilbert"

    # MGSD 
    mgsd_trained_distil = train_model(
        train_data=train_data_mgsd,
        model_path="distilbert/distilbert-base-uncased",
        batch_size=64,
        epoch=6,
        learning_rate=2e-5,
        model_output_base_dir=distil_model_base,
        dataset_name="mgsd_trained",
        seed=42,
    )
    evaluate_model(test_data_winoqueer_gpt_augmentation, mgsd_trained_distil, distil_results_base + "/mgsd_trained", "winoqueer_gpt_augmentation", 42)
    evaluate_model(test_data_seegull_gpt_augmentation, mgsd_trained_distil, distil_results_base + "/mgsd_trained", "seegull_gpt_augmentation", 42)
    evaluate_model(test_data_mgsd, mgsd_trained_distil, distil_results_base + "/mgsd_trained", "mgsd", 42)
    evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, mgsd_trained_distil, distil_results_base + "/mgsd_trained", "merged_winoqueer_seegull_gpt_augmentation", 42)

    # WinoQueer 
    wino_trained_distil = train_model(
        train_data=train_data_winoqueer_gpt_augmentation,
        model_path="distilbert/distilbert-base-uncased",
        batch_size=64,
        epoch=6,
        learning_rate=2e-5,
        model_output_base_dir=distil_model_base,
        dataset_name="winoqueer_gpt_augmentation_trained",
        seed=42,
    )
    evaluate_model(test_data_winoqueer_gpt_augmentation, wino_trained_distil, distil_results_base + "/winoqueer_gpt_augmentation_trained", "winoqueer_gpt_augmentation", 42)
    evaluate_model(test_data_seegull_gpt_augmentation, wino_trained_distil, distil_results_base + "/winoqueer_gpt_augmentation_trained", "seegull_gpt_augmentation", 42)
    evaluate_model(test_data_mgsd, wino_trained_distil, distil_results_base + "/winoqueer_gpt_augmentation_trained", "mgsd", 42)
    evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, wino_trained_distil, distil_results_base + "/winoqueer_gpt_augmentation_trained", "merged_winoqueer_seegull_gpt_augmentation", 42)

    # SeeGULL 
    seegull_trained_distil = train_model(
        train_data=train_data_seegull_gpt_augmentation,
        model_path="distilbert/distilbert-base-uncased",
        batch_size=64,
        epoch=6,
        learning_rate=2e-5,
        model_output_base_dir=distil_model_base,
        dataset_name="seegull_gpt_augmentation_trained",
        seed=42,
    )
    evaluate_model(test_data_winoqueer_gpt_augmentation, seegull_trained_distil, distil_results_base + "/seegull_gpt_augmentation_trained", "winoqueer_gpt_augmentation", 42)
    evaluate_model(test_data_seegull_gpt_augmentation, seegull_trained_distil, distil_results_base + "/seegull_gpt_augmentation_trained", "seegull_gpt_augmentation", 42)
    evaluate_model(test_data_mgsd, seegull_trained_distil, distil_results_base + "/seegull_gpt_augmentation_trained", "mgsd", 42)
    evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, seegull_trained_distil, distil_results_base + "/seegull_gpt_augmentation_trained", "merged_winoqueer_seegull_gpt_augmentation", 42)

    # Merged
    merged_trained_distil = train_model(
        train_data=train_data_merged_winoqueer_seegull_gpt_augmentation,
        model_path="distilbert/distilbert-base-uncased",
        batch_size=64,
        epoch=6,
        learning_rate=2e-5,
        model_output_base_dir=distil_model_base,
        dataset_name="merged_winoqueer_seegull_gpt_augmentation_trained",
        seed=42,
    )
    evaluate_model(test_data_winoqueer_gpt_augmentation, merged_trained_distil, distil_results_base + "/merged_winoqueer_seegull_gpt_augmentation_trained", "winoqueer_gpt_augmentation", 42)
    evaluate_model(test_data_seegull_gpt_augmentation, merged_trained_distil, distil_results_base + "/merged_winoqueer_seegull_gpt_augmentation_trained", "seegull_gpt_augmentation", 42)
    evaluate_model(test_data_mgsd, merged_trained_distil, distil_results_base + "/merged_winoqueer_seegull_gpt_augmentation_trained", "mgsd", 42)
    evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, merged_trained_distil, distil_results_base + "/merged_winoqueer_seegull_gpt_augmentation_trained", "merged_winoqueer_seegull_gpt_augmentation", 42)

    # BERT 
    bert_model_base = "model_output_bert"
    bert_results_base = "result_output_bert"

    # MGSD 
    mgsd_trained_bert = train_model(
        train_data=train_data_mgsd,
        model_path="google-bert/bert-base-uncased",
        batch_size=64,
        epoch=6,
        learning_rate=2e-5,
        model_output_base_dir=bert_model_base,
        dataset_name="mgsd_trained",
        seed=42,
    )
    evaluate_model(test_data_winoqueer_gpt_augmentation, mgsd_trained_bert, bert_results_base + "/mgsd_trained", "winoqueer_gpt_augmentation", 42)
    evaluate_model(test_data_seegull_gpt_augmentation, mgsd_trained_bert, bert_results_base + "/mgsd_trained", "seegull_gpt_augmentation", 42)
    evaluate_model(test_data_mgsd, mgsd_trained_bert, bert_results_base + "/mgsd_trained", "mgsd", 42)
    evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, mgsd_trained_bert, bert_results_base + "/mgsd_trained", "merged_winoqueer_seegull_gpt_augmentation", 42)

    # WinoQueer 
    wino_trained_bert = train_model(
        train_data=train_data_winoqueer_gpt_augmentation,
        model_path="google-bert/bert-base-uncased",
        batch_size=64,
        epoch=6,
        learning_rate=2e-5,
        model_output_base_dir=bert_model_base,
        dataset_name="winoqueer_gpt_augmentation_trained",
        seed=42,
    )
    evaluate_model(test_data_winoqueer_gpt_augmentation, wino_trained_bert, bert_results_base + "/winoqueer_gpt_augmentation_trained", "winoqueer_gpt_augmentation", 42)
    evaluate_model(test_data_seegull_gpt_augmentation, wino_trained_bert, bert_results_base + "/winoqueer_gpt_augmentation_trained", "seegull_gpt_augmentation", 42)
    evaluate_model(test_data_mgsd, wino_trained_bert, bert_results_base + "/winoqueer_gpt_augmentation_trained", "mgsd", 42)
    evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, wino_trained_bert, bert_results_base + "/winoqueer_gpt_augmentation_trained", "merged_winoqueer_seegull_gpt_augmentation", 42)

    # SeeGULL 
    seegull_trained_bert = train_model(
        train_data=train_data_seegull_gpt_augmentation,
        model_path="google-bert/bert-base-uncased",
        batch_size=64,
        epoch=6,
        learning_rate=2e-5,
        model_output_base_dir=bert_model_base,
        dataset_name="seegull_gpt_augmentation_trained",
        seed=42,
    )
    evaluate_model(test_data_winoqueer_gpt_augmentation, seegull_trained_bert, bert_results_base + "/seegull_gpt_augmentation_trained", "winoqueer_gpt_augmentation", 42)
    evaluate_model(test_data_seegull_gpt_augmentation, seegull_trained_bert, bert_results_base + "/seegull_gpt_augmentation_trained", "seegull_gpt_augmentation", 42)
    evaluate_model(test_data_mgsd, seegull_trained_bert, bert_results_base + "/seegull_gpt_augmentation_trained", "mgsd", 42)
    evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, seegull_trained_bert, bert_results_base + "/seegull_gpt_augmentation_trained", "merged_winoqueer_seegull_gpt_augmentation", 42)

    # Merged
    merged_trained_bert = train_model(
        train_data=train_data_merged_winoqueer_seegull_gpt_augmentation,
        model_path="google-bert/bert-base-uncased",
        batch_size=64,
        epoch=6,
        learning_rate=2e-5,
        model_output_base_dir=bert_model_base,
        dataset_name="merged_winoqueer_seegull_gpt_augmentation_trained",
        seed=42,
    )
    evaluate_model(test_data_winoqueer_gpt_augmentation, merged_trained_bert, bert_results_base + "/merged_winoqueer_seegull_gpt_augmentation_trained", "winoqueer_gpt_augmentation", 42)
    evaluate_model(test_data_seegull_gpt_augmentation, merged_trained_bert, bert_results_base + "/merged_winoqueer_seegull_gpt_augmentation_trained", "seegull_gpt_augmentation", 42)
    evaluate_model(test_data_mgsd, merged_trained_bert, bert_results_base + "/merged_winoqueer_seegull_gpt_augmentation_trained", "mgsd", 42)
    evaluate_model(test_data_merged_winoqueer_seegull_gpt_augmentation, merged_trained_bert, bert_results_base + "/merged_winoqueer_seegull_gpt_augmentation_trained", "merged_winoqueer_seegull_gpt_augmentation", 42)


if __name__ == "__main__":
    # For the full HEARTS pipeline, call run_full_hearts_pipeline() from a notebook.
    print("Running ALBERT baseline on MGSD ...")
    train_mgsd, test_mgsd = data_loader(
        csv_file_path="MGSD.csv",
        labelling_criteria="stereotype",
        dataset_name="MGSD",
        sample_size=1_000_000,
        num_examples=5,
    )
    model_output = train_model(
        train_data=train_mgsd,
        model_path="albert/albert-base-v2",
        batch_size=64,
        epoch=3,    # shorter for baseline
        learning_rate=2e-5,
        model_output_base_dir="model_output_albertv2",
        dataset_name="mgsd_baseline",
        seed=42,
    )
    evaluate_model(
        test_data=test_mgsd,
        model_output_dir=model_output,
        result_output_base_dir="results_albertv2",
        dataset_name="mgsd",
        seed=42,
    )
