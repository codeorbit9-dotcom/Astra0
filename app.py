import os
import pickle
import gdown
from flask import Flask, request, jsonify
from vosk import Model, KaldiRecognizer
import wave
import json

app = Flask(__name__)

# Ensure models folder exists
MODEL_DIR = "models"
VOSK_DIR = "vosk-model-small-en-in-0.4"
os.makedirs(MODEL_DIR, exist_ok=True)

# Download classifier.pkl
if not os.path.exists(f"{MODEL_DIR}/classifier.pkl"):
    gdown.download(
        "https://drive.google.com/uc?id=1tECA3f8zEkxeOEryWF66lWZHWeOUvT9Z",
        f"{MODEL_DIR}/classifier.pkl",
        quiet=False
    )

# Download vectorizer.pkl
if not os.path.exists(f"{MODEL_DIR}/vectorizer.pkl"):
    gdown.download(
        "https://drive.google.com/uc?id=1mNVj74l7ilV88WQV81BTntC-n7EcI3JQ",
        f"{MODEL_DIR}/vectorizer.pkl",
        quiet=False
    )

# Load ML models
with open(f"{MODEL_DIR}/classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

with open(f"{MODEL_DIR}/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Vosk STT download placeholder
if not os.path.exists(VOSK_DIR):
    print("Download Vosk model manually to Google Drive and update link if needed")

# Simple API for testing
@app.route("/")
def index():
    return jsonify({"status": "Astra Space running"})

# Classify text
@app.route("/classify", methods=["POST"])
def classify_text():
    text = request.json.get("text", "")
    vect = vectorizer.transform([text])
    prediction = classifier.predict(vect)[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
