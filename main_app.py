import os
import librosa
import numpy as np
import speech_recognition as sr
import streamlit as st
import tensorflow as tf

# Optional audio recorder component for live mic recording
try:
  from audiorecorder import audiorecorder

  HAS_AUDIORECORDER = True
except ImportError:
  HAS_AUDIORECORDER = False

st.set_page_config(
    page_title="Audio Speech-to-Text & Sentiment Analysis", layout="centered"
)

st.title("🎙️ Audio Speech-to-Text & Sentiment Analysis (Tiny ANN)")
st.write(
    "Record your voice or upload an audio file to transcribe speech and"
    " classify its emotional sentiment using a Tiny TFLite ANN model."
)


# Load TFLite Model with caching
@st.cache_resource
def load_tflite_model():
  model_path = "model/audio_sentiment_tiny.tflite"
  if not os.path.exists(model_path):
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

# Choose Input Method: Upload vs. Live Recording
input_method = st.radio(
    "Select Audio Input Method:", ("Upload Audio File", "Record Live Audio")
)

audio_path = "temp_audio.wav"
audio_ready = False

if input_method == "Upload Audio File":
  uploaded_file = st.file_uploader(
      "Upload Audio File (.wav or .mp3)", type=["wav", "mp3"]
  )
  if uploaded_file is not None:
    with open(audio_path, "wb") as f:
      f.write(uploaded_file.getbuffer())
    audio_ready = True
    st.audio(audio_path, format="audio/wav")

else:
  st.write("Click record below to capture audio directly from your microphone:")
  if HAS_AUDIORECORDER:
    recorded_audio = audiorecorder("Start Recording", "Stop Recording")
    if len(recorded_audio) > 0:
      # Export recorded audio to wav file
      recorded_audio.export(audio_path, format="wav")
      audio_ready = True
      st.audio(audio_path, format="audio/wav")
  else:
    st.info(
        "💡 To enable live mic recording, add `streamlit-audiorecorder` to your"
        " requirements.txt. Alternatively, use the file uploader above!"
    )
    # Standard fallback streamlit file uploader if audiorecorder package is absent
    fallback_file = st.file_uploader(
        "Upload recorded .wav file", type=["wav", "mp3"]
    )
    if fallback_file is not None:
      with open(audio_path, "wb") as f:
        f.write(fallback_file.getbuffer())
      audio_ready = True
      st.audio(audio_path, format="audio/wav")

# Analysis execution section
if audio_ready and st.button("🚀 Process Audio & Analyze Sentiment"):

  # Create an expandable container explaining step-by-step what is happening
  with st.status(
      "⚙️ Executing pipeline steps...", expanded=True
  ) as status_container:

    st.write(
        "📁 **Step 1: File Ingestion** — Captured audio file stored securely in"
        " temporary runtime environment."
    )

    # 1. Speech-to-Text Transcription
    st.write(
        "📝 **Step 2: Speech-to-Text Transcription** — Passing waveform to Google"
        " Speech Recognition API..."
    )
    recognizer = sr.Recognizer()
    transcript = ""
    try:
      with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        transcript = recognizer.recognize_google(audio_data)
      st.write(f'   ↳ *Transcription successful:* "{transcript}"')
    except Exception as sr_error:
      transcript = f"[Transcription Note: {str(sr_error)}]"
      st.write(f"   ↳ *Transcription warning:* {str(sr_error)}")

    # 2. Audio Feature Extraction (MFCCs)
    st.write(
        "🔬 **Step 3: Feature Extraction (Preprocessing)** — Loading audio with"
        " Librosa (22050Hz) and extracting 40 Mel-Frequency Cepstral Coefficients"
        " (MFCCs) as a mathematical fingerprint."
    )
    try:
      audio, sr_rate = librosa.load(audio_path, duration=3.0, sr=22050)
      mfccs = librosa.feature.mfcc(y=audio, sr=sr_rate, n_mfcc=40)
      scaled_mfccs = np.mean(mfccs.T, axis=0).astype(np.float32)
      input_data = np.expand_dims(scaled_mfccs, axis=0)
      st.write(
          f"   ↳ *Extracted feature shape vector:* {input_data.shape} (Ready"
          " for model input)"
      )

      # 3. Tiny ANN TFLite Inference
      st.write(
          "🧠 **Step 4: Tiny ANN Inference** — Feeding feature vector into"
          " optimized `audio_sentiment_tiny.tflite` model interpreter..."
      )
      interpreter.set_tensor(input_details[0]["index"], input_data)
      interpreter.invoke()
      output_data = interpreter.get_tensor(output_details[0]["index"])

      predicted_class = np.argmax(output_data[0])
      confidence = float(np.max(output_data[0]))
      st.write(
          "   ↳ *Inference complete!* Raw probabilities calculated successfully."
      )

      status_container.update(
          label="✅ Pipeline execution completed successfully!",
          state="complete",
          expanded=False,
      )

    except Exception as model_error:
      status_container.update(
          label="❌ Pipeline execution failed.", state="error", expanded=True
      )
      st.error(f"Inference execution failed: {str(model_error)}")
      stopped = True

  # Display Results Outside the Status Box
  if "transcript" in locals() and "output_data" in locals():
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
      st.subheader("📝 Speech Transcription")
      st.success(
          transcript if transcript else "No speech detected or transcribed."
      )

    with col2:
      st.subheader("🧠 Sentiment Prediction")
      # Map classes to human-readable emotions (adjust based on your dataset LabelEncoder classes)
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

      st.metric(
          label="Predicted Emotion",
          value=sentiment_result,
          delta=f"Confidence: {confidence * 100:.2f}%",
      )

# Cleanup temporary file on rerun
if os.path.exists(audio_path) and not audio_ready:
  pass
