import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from collections import deque

# ---------------- Load Model ----------------
<<<<<<< HEAD
model = tf.keras.models.load_model("emotion_model.h5")
=======
MODEL_PATH= "emotion_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
>>>>>>> effe2b1ece0db44f86fd7bc9ec9fc401c737af8c
st.write(f"Model loaded. Expected input shape: {model.input_shape}")

# Determine expected channels
_, h, w, c = model.input_shape
is_grayscale = (c == 1)

# Define emotion classes
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
CONFIDENCE_THRESHOLD = 0.6  # highlight low-confidence predictions

st.title("Emotion Recognition App")
input_type = st.sidebar.selectbox("Input type", ["Upload Image", "Webcam"])

# ---------------- Helper Functions ----------------
def preprocess_image(img_pil):
    if is_grayscale:
        img_pil = img_pil.convert("L")
    img_resized = img_pil.resize((w, h))
    img_array = np.array(img_resized).astype(np.float32)
    if is_grayscale:
        img_array = np.expand_dims(img_array, axis=-1)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    return preprocess_image(img_pil)

def predict_emotion(img_array):
    preds = model.predict(img_array)
    return preds[0]

def get_label_color(conf):
    return 'red' if conf < CONFIDENCE_THRESHOLD else 'green'

# ---------------- Upload Image ----------------
if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg","png","jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess_image(img)
        preds = predict_emotion(img_array)
        pred_idx = np.argmax(preds)
        confidence = preds[pred_idx]

        color = get_label_color(confidence)
        st.markdown(
            f"Predicted Emotion: <span style='color:{color};'>**{emotion_classes[pred_idx]}** ({confidence:.2f})</span>",
            unsafe_allow_html=True
        )

        st.bar_chart({emotion_classes[i]: float(preds[i]) for i in range(len(emotion_classes))})

# ---------------- Webcam ----------------
else:
    st.write("Webcam live feed (press Start Webcam to begin)")
    run = st.checkbox("Start Webcam")  # define once

    if run:
        cap = cv2.VideoCapture(0)
        pred_history = deque(maxlen=5)

        # Create a container to hold frame and chart
        webcam_container = st.container()
        frame_placeholder = webcam_container.empty()
        chart_placeholder = webcam_container.empty()

        try:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Could not read frame from webcam")
                    break

                # Preprocess frame and predict
                img_array = preprocess_frame(frame)
                preds = predict_emotion(img_array)
                pred_history.append(preds)
                avg_preds = np.mean(pred_history, axis=0)
                pred_idx = np.argmax(avg_preds)
                confidence = avg_preds[pred_idx]

                # Draw label on frame
                color = (0,0,255) if confidence < CONFIDENCE_THRESHOLD else (0,255,0)
                label = f"{emotion_classes[pred_idx]} ({confidence:.2f})"
                cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Update frame and single bar chart in container
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                chart_placeholder.bar_chart(
                    {emotion_classes[i]: float(avg_preds[i]) for i in range(len(emotion_classes))}
                )

                # Keep checkbox state synced
                run = st.session_state.get("Start Webcam", run)

        finally:
            cap.release()
