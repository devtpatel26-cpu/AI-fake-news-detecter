# Academic Fake News Detector

A complete web-based fake news detection project using a Python FastAPI backend and an HTML/CSS/JavaScript frontend.

## Files

- `app.py`: FastAPI backend that loads DistilBERT and serves prediction requests.
- `templates/index.html`: Static web page for the frontend.
- `static/style.css`: Styling for the web interface.
- `static/app.js`: Frontend JavaScript for user interaction and API calls.
- `requirements.txt`: Python dependencies.
- `deep_learning_app.py`: Optional existing Streamlit demo.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run

Start the backend with:

```bash
uvicorn app:app --reload
```

Then open the frontend in your browser at:

```text
http://127.0.0.1:8000
```

## Notes

- The backend trains on synthetic fake news samples at startup for demonstration purposes.
- The frontend is implemented using static HTML, CSS, and JavaScript.
- The model uses Hugging Face `distilbert-base-uncased` and PyTorch.
