# ==============================================================
# 🧠 SPEECH-BASED GENDER DETECTION (Local VS Code)
# With Live Microphone Recording
# ==============================================================

import os
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# 1️⃣ Feature Extraction
# -----------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    return np.hstack([mfccs, chroma, zcr, spec_cent])

# -----------------------------
# 2️⃣ Record Voice Function
# -----------------------------
def record_voice(filename="recorded_sample.wav", duration=3, sr=16000):
    """
    Records audio from the microphone and saves it as a WAV file.
    """
    print(f"🎤 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    sf.write(filename, audio, sr)
    print(f"✅ Recording saved as {filename}")
    return filename

# -----------------------------
# 3️⃣ Build Dataset
# -----------------------------
dataset_folder = "dataset"  # Folder containing male_*.wav and female_*.wav
X, y = [], []

for fname in tqdm(os.listdir(dataset_folder)):
    if fname.endswith(".wav"):
        path = os.path.join(dataset_folder, fname)
        label = 0 if fname.lower().startswith("male") else 1
        try:
            feats = extract_features(path)
            X.append(feats)
            y.append(label)
        except Exception as e:
            print("⚠️ Error:", path, e)

X, y = np.array(X), np.array(y)
print(f"\n✅ Loaded {len(X)} samples")

# -----------------------------
# 4️⃣ Check Class Balance
# -----------------------------
unique_classes = np.unique(y)
if len(unique_classes) < 2:
    raise ValueError(
        f"❌ Not enough classes to train! Found only {len(unique_classes)} class. "
        "Upload at least one male and one female audio file."
    )

# -----------------------------
# 5️⃣ Train/Test Split & Scaling
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# -----------------------------
# 6️⃣ Train SVM Model
# -----------------------------
model = SVC(kernel='rbf', C=10, gamma=0.1)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\n✅ Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, pred, target_names=["Male", "Female"]))

joblib.dump(model, "gender_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model saved successfully!")
# -----------------------------
# 7️⃣ Predict Function
# -----------------------------
def predict_audio(file_path):
    feats = extract_features(file_path)
    feats = scaler.transform([feats])
    pred = model.predict(feats)[0]
    return "Male 🎤" if pred == 0 else "Female 🎤"

# -----------------------------
# 8️⃣ Record & Predict Live Voice
# -----------------------------
test_path = record_voice(filename="my_voice.wav", duration=5)
print("Prediction:", predict_audio(test_path))

# -----------------------------
# 9️⃣ Visualize MFCC of Recorded Sample
# -----------------------------
y_audio, sr = librosa.load(test_path)
mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
plt.figure(figsize=(8,4))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title('MFCC of Recorded Sample')
plt.tight_layout()
plt.show()
