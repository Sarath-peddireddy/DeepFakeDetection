# Deepfake Detection Project

## Overview
This project aims to detect **Deepfake videos** using machine learning models. Deepfakes are synthetic media where someone's likeness is digitally manipulated, and detecting these videos is crucial for maintaining authenticity in media content.

In this project, I have developed and trained a machine learning model to distinguish between real and fake videos by analyzing facial movements and other key indicators. The model has been trained on a subset of a larger dataset and is able to achieve good performance for detecting face-swapping in videos.

## Dataset
The dataset used in this project is a subset of the **[FaceForensics++](https://github.com/ondyari/FaceForensics) dataset**, which contains manipulated videos for training and evaluation. This dataset is widely used in research for detecting facial manipulations and deepfakes.

### Subset Details:
- **Real Videos:** Authentic, unaltered video samples.
- **Fake Videos:** Videos manipulated with deepfake techniques (face-swapping).
- **Size:** A reduced version of the full dataset was used due to memory constraints.
- **Format:** The videos are in `.mp4` format with annotations labeling each video as real or fake.

## Model and Training
The model used for this project is a **Convolutional Neural Network (CNN)**, fine-tuned for deepfake detection. Key aspects of the model training:
- **Pretrained Weights:** The model was initialized with weights from a pretrained network, enabling faster convergence on this specific task.
- **Training Setup:** Trained on a subset of the **FaceForensics++** dataset.
- **Memory Constraints:** Due to GitHub's file size limitations, the full pretrained model exceeds the 25MB upload limit, so you will need to download the pretrained weights externally.
- **Framework:** The model was built using **PyTorch** and leverages popular deep learning libraries for video processing.

## Results
The trained model shows strong performance on the test dataset, detecting deepfakes with a high level of accuracy.
## Usage

To run the model and detect deepfakes on your videos, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/Sarath-peddireddy/DeepFakeDetection
cd DeepFakeDetection
