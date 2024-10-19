# Deepfake Detection App

This project implements a Deepfake detection application using TensorFlow and Streamlit. The application analyzes uploaded video files to predict whether they are real or fake.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Features
- Upload a video and receive a prediction on whether it's real or fake.
- Displays the uploaded video within the app.
- Outputs the confidence of the prediction.

## Technologies Used
- **TensorFlow**: For building and training the deep learning model.
- **OpenCV**: For video processing and face detection.
- **face_recognition**: For accurate face detection.
- **Streamlit**: To create a user-friendly web interface.
- **NumPy**: For numerical operations.

## Installation
To run this application, ensure you have Python installed (preferably Python 3.7 or later). Then, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/<repo-name>.git
   cd <repo-name>
