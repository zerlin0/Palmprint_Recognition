# Contactless Palmprint Recognition System

This project implements a contactless palmprint recognition system using deep learning (ResNet50), trained in two modes: fine-tuned from ImageNet and from scratch. The system is deployed with a Streamlit-based web interface for real-time palmprint identification.

## 🔍 Features

- Contactless palmprint recognition
- Fine-tuned and scratch-trained ResNet50 comparison
- ROI extraction using YOLOv8
- Data augmentation (brightness, contrast, tilt, CLAHE)
- Streamlit web interface for testing
- Per-class metrics and confusion matrix analysis
- CSV result export and clear log function

## 🧠 Model Architectures

- **ResNet50 (Fine-Tuned)**: Initialized with ImageNet weights and retrained on palmprints.
- **ResNet50 (Scratch)**: Random weight initialization, trained entirely on palmprints.

## 🧪 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Per-class performance
