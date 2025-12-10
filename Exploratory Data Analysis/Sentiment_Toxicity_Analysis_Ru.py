import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import torch


# Russian Sentiment Classifier (Blanchefort)
class RussianSentimentClassifier:
    """
    Sentiment on Russian text using blanchefort/rubert-base-cased-sentiment.

    Label mapping:
    - 0 -> negative
    - 1 -> neutral
    - 2 -> positive
    """

    def __init__(self):
        model_name = "blanchefort/rubert-base-cased-sentiment"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, return_dict=True
        )

        # GPU / CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # label mapping for readability
        self.label_map = {
            0: "neutral",
            1: "positive",
            2: "negative",
        }

    @torch.no_grad()
    def predict_single(self, text: str) -> str:
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(torch.nn.functional.softmax(logits, dim=1), dim=1).item()

        return self.label_map.get(pred_id, str(pred_id))

    def predict_batch(self, texts):
        return [self.predict_single(t) for t in texts]


# Russian Toxicity / "Regard" Classifier
class RussianToxicityClassifier:
    """
    Toxicity classifier using sismetanin/rubert-toxic-pikabu-2ch.
    Used as a proxy for negative regard / harm.

    Label mapping:
    - LABEL_1 -> toxic
    - LABEL_0 -> non_toxic
    """

    def __init__(self):
        model_name = "sismetanin/rubert-toxic-pikabu-2ch"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        device = 0 if torch.cuda.is_available() else -1

        self.pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            truncation=True,
            max_length=512,
        )

        self.label_map = {
            "LABEL_1": "toxic",
            "LABEL_0": "non_toxic",
        }

    def predict(self, texts):
        outputs = self.pipe(texts, batch_size=16)
        return [self.label_map.get(o["label"], o["label"]) for o in outputs]


# Combined Analysis Function
def analyse_sentiment_and_regard(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    
    """
    Add Russian sentiment (blanchefort) and toxicity-based regard (sismetanin) to the dataframe.

    Columns added:
    - Sentiment_ru : {'positive','neutral','negative'}
    - Regard_ru    : {'toxic','non_toxic'}
    """

    df = df.copy()

    sent_clf = RussianSentimentClassifier()
    tox_clf = RussianToxicityClassifier()

    df["sentiment"] = [
        sent_clf.predict_single(text)
        for text in tqdm(df[text_col], desc="Analyzing Russian Sentiment")
    ]

    df["regard"] = [
        tox_clf.predict([text])[0]
        for text in tqdm(df[text_col], desc="Analyzing Russian Toxicity / Regard")
    ]

    return df


# Optional CLI usage
if __name__ == "__main__":
    df_example = pd.read_csv("COMP0173_Data/rubist.csv")
    df_scored = analyse_sentiment_and_regard(df_example, text_col="text")
    df_scored.to_csv("rubist_sentiment_regard_ru_blanch.csv", index=False)