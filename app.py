from flask import Flask, render_template, request
import joblib
import numpy as np
import librosa
import os
import soundfile as sf
import pydub as AudioSegment


app = Flask(__name__)

# Load model + scaler
model = joblib.load("gender_model.pkl")
scaler = joblib.load("scaler.pkl")


# Feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    return np.hstack([mfccs, chroma, zcr, spec_cent])


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        audio = request.files["audio_file"]

        # Save temporary file
        temp = "static/temp_audio"
        audio.save(temp)

        # Convert to real wav
        from pydub import AudioSegment
        sound = AudioSegment.from_file(temp)
        sound = sound.set_frame_rate(16000).set_channels(1)
        final_path = "static/last_audio.wav"
        sound.export(final_path, format="wav")

        # Extract features
        feats = extract_features(final_path)
        feats = scaler.transform([feats])
        pred = model.predict(feats)[0]

        result = "Male" if pred == 0 else "Female"

    return render_template("index.html", result=result)



if __name__ == "__main__":
    app.run(debug=True)
