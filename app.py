import os
import pickle
import requests
from flask import Flask, request, jsonify
from vosk import Model, KaldiRecognizer
import wave
import pyttsx3

# --- Config ---
CLASSIFIER_URL = "https://drive.google.com/uc?id=1tECA3f8zEkxeOEryWF66lWZHWeOUvT9Z&export=download"
VECTORIZER_URL = "https://drive.google.com/uc?id=1mNVj74l7ilV88WQV81BTntC-n7EcI3JQ&export=download"

os.makedirs("models", exist_ok=True)
CLASSIFIER_PATH = "models/classifier.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

# --- Download models if missing ---
def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading {path} ...")
        r = requests.get(url, stream=True)
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"{path} downloaded.")

download_file(CLASSIFIER_URL, CLASSIFIER_PATH)
download_file(VECTORIZER_URL, VECTORIZER_PATH)

# --- Load models ---
with open(CLASSIFIER_PATH, "rb") as f:
    classifier = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# --- Setup Flask ---
app = Flask(__name__)

# --- Initialize TTS ---
engine = pyttsx3.init()

# --- STT using Vosk ---
# Make sure you have downloaded vosk-model-small-en-in-0.4 manually or hosted it externally
if not os.path.exists("vosk-model-small-en-in-0.4"):
    raise FileNotFoundError("Please download vosk-model-small-en-in-0.4 manually.")

vosk_model = Model("vosk-model-small-en-in-0.4")

@app.route("/")
def index():
    return jsonify({"status": "AI Guardian running"})

@app.route("/stt", methods=["POST"])
def stt_endpoint():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]
    file_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    wf = wave.open(file_path, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())

    text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            text += result
    text += rec.FinalResult()
    wf.close()
    os.remove(file_path)

    return jsonify({"text": text})

@app.route("/classify", methods=["POST"])
def classify_endpoint():
    if "text" not in request.json:
        return jsonify({"error": "No text provided"}), 400
    user_text = request.json["text"]
    features = vectorizer.transform([user_text])
    prediction = classifier.predict(features)[0]
    return jsonify({"prediction": prediction})

@app.route("/tts", methods=["POST"])
def tts_endpoint():
    if "text" not in request.json:
        return jsonify({"error": "No text provided"}), 400
    user_text = request.json["text"]
    audio_file = f"tts_{hash(user_text)}.mp3"
    engine.save_to_file(user_text, audio_file)
    engine.runAndWait()
    return jsonify({"audio_path": audio_file})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=7860)
