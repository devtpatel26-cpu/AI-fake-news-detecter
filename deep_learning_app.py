"""
Academic Fake News Detector

Run:
    pip install streamlit transformers torch pandas scikit-learn
    streamlit run deep_learning_app.py
"""

import random
from typing import Dict, Tuple

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, DistilBertModel, DistilBertTokenizerFast


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


@st.cache_data(show_spinner=False)
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


@st.cache_resource(show_spinner=False)
def load_tokenizer() -> DistilBertTokenizerFast:
    return DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


@st.cache_resource(show_spinner=False)
def build_model() -> FakeNewsClassifier:
    return FakeNewsClassifier()


@st.cache_data(show_spinner=False)
def train_model(tokenizer: DistilBertTokenizerFast, device_str: str) -> Tuple[FakeNewsClassifier, pd.DataFrame]:
    device = torch.device(device_str)
    df = generate_synthetic_news(num_samples=50)
    train_df, _ = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train_dataset = FakeNewsDataset(train_df["text"], train_df["label"], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = build_model().to(device)
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
        print(f"[Training] Epoch 1 finished. Average loss: {avg_loss:.4f}")

    return model, df


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
        "probability_fake": prob,
        "probability_real": 1.0 - prob,
        "prediction": "Fake" if prob >= 0.5 else "Real",
    }


def render_probability_gauge(probability: float) -> None:
    percent = probability * 100
    gauge_style = f"width: {percent:.1f}%;"
    st.markdown(
        f"""
        <style>
            .gauge-wrapper {{
                background: #f1f1f1;
                border-radius: 12px;
                overflow: hidden;
                margin: 16px 0;
            }}
            .gauge-fill {{
                background: linear-gradient(90deg, #b31217, #e52d27);
                height: 24px;
                width: 0;
                border-radius: 12px;
                transition: width 0.8s ease-in-out;
            }}
            .gauge-label {{
                font-weight: 600;
                margin-top: 8px;
                display: block;
            }}
        </style>
        <div class="gauge-wrapper">
            <div class="gauge-fill" style="{gauge_style}"></div>
        </div>
        <span class="gauge-label">Confidence: {percent:.1f}% fake news probability</span>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer()

    st.set_page_config(
        page_title="Academic Fake News Detector",
        page_icon="📰",
        layout="wide",
    )

    st.title("Academic Fake News Detector")
    st.write(
        "This Streamlit demo fine-tunes DistilBERT on synthetic fake news samples, then classifies new text in real time."
    )

    with st.sidebar:
        st.header("How to use this app")
        st.write(
            "1. Paste any news text or headline.\n"
            "2. Click the button to classify.\n"
            "3. View the confidence gauge and model output."
        )
        st.markdown("---")
        st.subheader("Project files")
        st.write("- `deep_learning_app.py`\n- `requirements.txt`\n- `README.md`")

    status_placeholder = st.empty()
    with st.spinner("Loading model and preparing data..."):
        model, training_df = train_model(tokenizer, device.type)
        status_placeholder.success("Model ready for inference.")

    left_col, right_col = st.columns([3, 1])

    with left_col:
        st.subheader("Enter text to classify")
        user_text = st.text_area(
            "Paste a news article or headline here:",
            height=220,
            placeholder="Enter a news sentence for fake news detection...",
        )

        if st.button("Analyze with Deep Learning"):
            if not user_text.strip():
                st.warning("Please enter a news article before analysis.")
            else:
                result = predict_news(model, tokenizer, user_text, device)
                st.success(f"Prediction: {result['prediction']}")
                st.write(
                    f"Model output indicates a **{result['probability_fake'] * 100:.1f}%** probability that the input is fake news."
                )
                render_probability_gauge(result["probability_fake"])
                st.markdown(
                    "### Prediction details\n"
                    f"- **Fake score:** {result['probability_fake']:.3f}\n"
                    f"- **Real score:** {result['probability_real']:.3f}\n"
                    f"- **Decision threshold:** 0.50\n"
                )

    with right_col:
        st.subheader("Synthetic training samples")
        st.dataframe(training_df.head(8), hide_index=True)
        st.markdown("---")
        st.subheader("Model explanation")
        st.write(
            "The custom `FakeNewsClassifier` wraps DistilBERT and uses the first token embedding from the last hidden layer. "
            "After dropout, a single linear layer outputs a binary logit for fake vs real classification."
        )
        classifier_weight_shape = tuple(model.classifier.weight.shape)
        st.write(f"Classifier weight shape: {classifier_weight_shape}")

    st.markdown("---")
    st.write(
        "**Note:** The app is intentionally designed for academic demonstration. "
        "It trains on synthetic text and does not use a real news dataset in order to run immediately."
    )


if __name__ == "__main__":
    main()
