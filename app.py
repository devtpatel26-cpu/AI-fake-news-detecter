from pathlib import Path
from typing import Dict

import pandas as pd
import random
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, DistilBertModel, DistilBertTokenizerFast

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Academic Fake News Detector",
    description="A web-based fake news classification demo using PyTorch and DistilBERT.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class PredictionRequest(BaseModel):
    text: str


class FakeNewsDataset(Dataset):
    def __init__(self, texts: pd.Series, labels: pd.Series, tokenizer: DistilBertTokenizerFast, max_length: int = 128):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels.tolist(), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: self.encodings[key][idx] for key in self.encodings}
        item["labels"] = self.labels[idx]
        return item


class FakeNewsClassifier(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", dropout_prob: float = 0.3):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        hidden_size = self.distilbert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def generate_synthetic_news(num_samples: int = 50) -> pd.DataFrame:
    random.seed(42)
    topics = [
        "international politics",
        "health and wellness",
        "technology breakthroughs",
        "celebrity gossip",
        "economic trends",
        "sports highlights",
        "science discoveries",
        "environmental alerts",
        "education policy",
        "cybersecurity news",
    ]
    actions = [
        "claims the government is hiding data",
        "reports a shocking new cure",
        "announces a breakthrough in AI",
        "warns of an imminent natural disaster",
        "says that a celebrity is secretly an alien",
        "reveals a hidden economic strategy",
        "describes the latest viral challenge",
        "explains a new scientific study",
        "predicts the collapse of a major currency",
        "criticizes a popular social media platform",
    ]
    labels = [0, 1]
    samples = []

    for _ in range(num_samples):
        topic = random.choice(topics)
        action = random.choice(actions)
        label = random.choice(labels)
        prefix = "Fact-check report:" if label == 0 else "Breaking update:"
        text = f"{prefix} A new story about {topic} {action}."
        samples.append({"text": text, "label": label})

    return pd.DataFrame(samples)


def train_model(tokenizer: DistilBertTokenizerFast, device: torch.device) -> FakeNewsClassifier:
    df = generate_synthetic_news(num_samples=50)
    train_df, _ = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train_dataset = FakeNewsDataset(train_df["text"], train_df["label"], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = FakeNewsClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(1):
        running_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"[Training] Finished epoch 1. Average loss: {avg_loss:.4f}")

    return model


def predict_news(model: FakeNewsClassifier, tokenizer: DistilBertTokenizerFast, text: str, device: torch.device) -> Dict[str, float]:
    model.eval()
    inputs = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        prob = torch.sigmoid(logits).cpu().item()

    return {
        "prediction": "Fake" if prob >= 0.5 else "Real",
        "probability_fake": round(prob, 4),
        "probability_real": round(1.0 - prob, 4),
    }


model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
def initialize_model() -> None:
    global model, tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = train_model(tokenizer, device)
    print("Model loaded and synthetic training completed.")


@app.get("/", response_class=FileResponse)
def read_index() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.post("/predict")
def predict(payload: PredictionRequest) -> JSONResponse:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet. Please wait and retry.")

    if not payload.text.strip():
        raise HTTPException(status_code=422, detail="Input text cannot be empty.")

    result = predict_news(model, tokenizer, payload.text, device)
    return JSONResponse(content=result)
