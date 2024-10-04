import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)
        features = np.hstack((mfccs_mean, chroma_mean, mel_mean))
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

data_dir = 'path/to/your/data'
genres = os.listdir(data_dir)

X = []
y = []

for genre in genres:
    genre_dir = os.path.join(data_dir, genre)
    if os.path.isdir(genre_dir):
        for filename in os.listdir(genre_dir):
            file_path = os.path.join(genre_dir, filename)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(genre)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
