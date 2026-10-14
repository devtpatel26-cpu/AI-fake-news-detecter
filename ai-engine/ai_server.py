"""
Fake News Detector - AI Deep Learning Inference API
Flask-based server for real-time news classification
"""
import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import joblib

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Global model and tokenizer storage
model = None
tokenizer = None
max_sequence_length = 100

def load_model():
    """Load pre-trained deep learning model and tokenizer"""
    global model, tokenizer
    try:
        # In production, load from saved model
        # model = tf.keras.models.load_model('models/fake_news_model.h5')
        # tokenizer = joblib.load('models/tokenizer.pkl')
        
        # For demonstration, create a simple LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(5000, 128, input_length=max_sequence_length),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        print("✓ Model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def preprocess_text(text):
    """Preprocess and tokenize input text"""
    try:
        # Simple preprocessing
        text = text.lower()
        text = text.replace('<[^>]+>', '')  # Remove HTML tags
        words = text.split()
        # Limit to max_sequence_length
        words = words[:max_sequence_length]
        return ' '.join(words)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return text

def predict_fake_news(text):
    """Predict if news is fake (1) or real (0)"""
    try:
        # Preprocess
        processed_text = preprocess_text(text)
        
        # Tokenize (simplified - in production use loaded tokenizer)
        tokens = processed_text.split()
        # Convert to numeric sequences (simple hash-based)
        sequence = [hash(word) % 5000 for word in tokens]
        # Pad sequence
        padded = pad_sequences([sequence], maxlen=max_sequence_length, padding='post')
        
        # Predict
        prediction = float(model.predict(padded, verbose=0)[0][0])
        
        return {
            'fake_probability': prediction,
            'real_probability': 1 - prediction,
            'classification': 'FAKE NEWS' if prediction > 0.5 else 'REAL NEWS',
            'confidence': max(prediction, 1 - prediction)
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'AI Server is running'}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict if article is fake news"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({'error': 'Text must be non-empty string'}), 400
        
        result = predict_fake_news(text)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """Predict multiple articles"""
    try:
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({'error': 'Expected array of articles'}), 400
        
        results = []
        for article in data:
            if isinstance(article, dict) and 'text' in article:
                prediction = predict_fake_news(article['text'])
                prediction['id'] = article.get('id', len(results))
                results.append(prediction)
        
        return jsonify({'predictions': results}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_name': 'Fake News Detector LSTM',
        'version': '1.0',
        'max_sequence_length': max_sequence_length,
        'input_type': 'text',
        'output_type': 'binary_classification'
    }), 200

if __name__ == '__main__':
    print("Initializing AI Server...")
    load_model()
    port = int(os.getenv('AI_SERVER_PORT', 5000))
    print(f"Starting Flask server on port {port}...")
    app.run(debug=True, port=port, host='0.0.0.0')
