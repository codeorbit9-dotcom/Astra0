import os
import requests
import zipfile
from vosk import Model, KaldiRecognizer

vosk_folder = "vosk-model-small-en-in-0.4"
zip_file = "vosk.zip"
drive_url = "https://drive.google.com/uc?export=download&id=1jjU6ZMFKoL4nN1smj5tV-YNicL5bNMx3"

if not os.path.exists(vosk_folder):
    print("Downloading Vosk model...")
    r = requests.get(drive_url, stream=True)
    with open(zip_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
    print("Extracting...")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(".")
    os.remove(zip_file)
    print("Vosk ready.")

model = Model(vosk_folder)