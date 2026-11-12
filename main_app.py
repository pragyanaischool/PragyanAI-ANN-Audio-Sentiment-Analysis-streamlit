import os
import librosa
import numpy as np
import speech_recognition as sr
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Audio Sentiment Analysis (Tiny ANN)", layout="centered"
)

st.title("🎙️ Audio Speech-to-Text & Sentiment Analysis")
st.write(
    "Upload an audio file to transcribe speech and classify sentiment using a"
    " Tiny TFLite ANN model."
)


# Load TFLite Model with caching
@st.cache_resource
def load_tflite_model():
  model_path = "model/audio_sentiment_tiny.tflite"
  if not os.path.exists(model_path):
    # Fallback to current directory if model folder is not structured
    model_path = "audio_sentiment_tiny.tflite"

  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()
  return interpreter


try:
  interpreter = load_tflite_model()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  st.sidebar.success("✅ Tiny TFLite Model Loaded Successfully!")
except Exception as e:
  st.sidebar.error(
      f"⚠️ TFLite model file not found or failed to load: {str(e)}"
  )

# Audio File Uploader
uploaded_file = st.file_uploader(
    "Upload Audio File (.wav or .mp3)", type=["wav", "mp3"]
)

if uploaded_file is not None:
  audio_path = "temp_audio.wav"
  with open(audio_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

  st.audio(audio_path, format="audio/wav")

  if st.button("Analyze Audio & Transcribe"):
    with st.spinner(
        "Processing speech transcription and sentiment classification..."
    ):

      # 1. Speech-to-Text Transcription
      recognizer = sr.Recognizer()
      transcript = ""
      try:
        with sr.AudioFile(audio_path) as source:
          audio_data = recognizer.record(source)
          transcript = recognizer.recognize_google(audio_data)
      except Exception as sr_error:
        transcript = (
            f"[Transcription Warning / Audio Format Note: {str(sr_error)}]"
        )

      st.subheader("📝 Transcription Result")
      st.success(transcript)

      # 2. Feature Extraction & ANN Inference
      try:
        audio, sr_rate = librosa.load(audio_path, duration=3.0, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr_rate, n_mfcc=40)
        scaled_mfccs = np.mean(mfccs.T, axis=0).astype(np.float32)
        input_data = np.expand_dims(scaled_mfccs, axis=0)

        # Run TFLite inference
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])

        predicted_class = np.argmax(output_data[0])
        confidence = float(np.max(output_data[0]))

        # Map classes to human-readable emotions (adjust based on your LabelEncoder classes)
        emotion_map = {
            0: "Neutral",
            1: "Calm",
            2: "Happy",
            3: "Sad",
            4: "Angry",
            5: "Fearful",
            6: "Disgusted",
            7: "Surprised",
        }
        sentiment_result = emotion_map.get(
            predicted_class, f"Class {predicted_class}"
        )

        st.subheader("🧠 Tiny ANN Sentiment Prediction")
        st.metric(
            label="Predicted Emotion / Sentiment",
            value=sentiment_result,
            delta=f"Confidence: {confidence * 100:.2f}%",
        )

      except Exception as model_error:
        st.error(f"Inference execution failed: {str(model_error)}")

  # Cleanup temporary file
  if os.path.exists("temp_audio.wav"):
    os.remove("temp_audio.wav")
