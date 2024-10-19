import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import face_recognition
import os
import random
import streamlit as st

# Define constants
num_classes = 2
seq_length = 20
frame_size = (112, 112)

# Load base model with transfer learning (using InceptionV3)
base_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=(112, 112, 3)
)

# Define the full model
model = models.Sequential([
    layers.Input(shape=(seq_length, *frame_size, 3)),
    layers.TimeDistributed(base_model),
    layers.TimeDistributed(layers.GlobalAveragePooling2D()),
    layers.LSTM(128, return_sequences=True),  # Increased LSTM units
    layers.LSTM(64),  # Adding another LSTM layer
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def augment_frame(frame):
    # Random horizontal flip
    if random.random() < 0.5:
        frame = np.fliplr(frame)

    # Random brightness adjustment
    if random.random() < 0.5:
        brightness_factor = 1.0 + random.uniform(-0.1, 0.1)
        frame = np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)

    return frame

def detect_deepfake(video_path):
    vidObj = cv2.VideoCapture(video_path)
    frames = []

    # Extract frames from the video
    while len(frames) < seq_length:
        success, frame = vidObj.read()
        if not success:
            break

        # Detect faces
        faces = face_recognition.face_locations(frame)
        if faces:
            top, right, bottom, left = faces[0]
            face_frame = frame[top:bottom, left:right, :]
            face_frame = cv2.resize(face_frame, frame_size)

            # Apply augmentation
            face_frame = augment_frame(face_frame)

            frames.append(face_frame)

    # Pad with zeros if fewer than sequence length
    if len(frames) < seq_length:
        frames += [np.zeros((112, 112, 3))] * (seq_length - len(frames))  # Padding

    # Stack frames to create input for the model
    input_data = np.stack(frames)  # Create a 4D tensor
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    # Make predictions
    logits = model.predict(input_data)
    prediction = np.argmax(logits, axis=1)[0]
    confidence = np.max(logits) * 100

    return prediction, confidence

# Streamlit Application
def main():
    st.title("Deepfake Detection App")
    st.write("Upload a video file to check if it's real or fake.")

    # Create directory if it doesn't exist
    if not os.path.exists("./uploaded_videos"):
        os.makedirs("./uploaded_videos")

    video_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        # Save the uploaded video file temporarily
        video_path = os.path.join("./uploaded_videos", video_file.name)

        # Check for the existence of the video file to avoid overwriting
        if os.path.exists(video_path):
            st.error("File already exists! Please choose a different file.")
        else:
            with open(video_path, "wb") as f:
                f.write(video_file.getbuffer())

            # Display the video in the app
            st.video(video_path)

            # Call the deepfake detection function
            prediction, confidence = detect_deepfake(video_path)

            # Clean up saved video file
            os.remove(video_path)

            # Format output based on prediction
            if prediction == 1:
                st.markdown(f"<h2 style='color: green;'>Prediction: REAL 👍</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color: red;'>Prediction: FAKE 👎</h2>", unsafe_allow_html=True)

            st.write(f"**Confidence of prediction: {confidence:.2f}%**")

if __name__ == "__main__":
    main()
