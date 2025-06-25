import numpy as np
import librosa
import tensorflow as tf
import sys
import os

# Load the model
model = tf.keras.models.load_model('emotion_classifier_gru.h5')

# Define emotion classes (in order of model output)
emotion_classes = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Feature extraction
def extract_features(file_path, max_len=200):
    y, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_len]
    return mfccs.T

# Predict emotion
def predict_emotion(audio_path):
    features = extract_features(audio_path)
    features = np.expand_dims(features, axis=0)  # (1, 200, 40)
    prediction = model.predict(features, verbose=0)[0]
    emotion = emotion_classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    return emotion, confidence

# Main: process multiple audio files
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_emotion.py file1.wav file2.wav ...")
        sys.exit(1)

    for audio_file in sys.argv[1:]:
        if not os.path.isfile(audio_file):
            print(f"❌ File not found: {audio_file}")
            continue

        try:
            emotion, confidence = predict_emotion(audio_file)
            print(f"🎧 {audio_file} → Predicted Emotion: {emotion} (Confidence: {confidence:.2f})")
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}")

