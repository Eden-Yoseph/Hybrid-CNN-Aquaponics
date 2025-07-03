# Hybrid-CNN-Aquaponics
# 🍅 Tomato Leaf Disease Classifier – Hybrid CNN Model

This project implements a hybrid convolutional neural network for binary classification of tomato leaf health using image data. It combines a custom-designed CNN with a pre-trained ResNet18 backbone to enhance performance in low-data conditions, achieving over **99.3% validation accuracy** and **93.8% F1-score**.

## 🔍 Objective

Tomato crops are vulnerable to a variety of leaf diseases, leading to major agricultural losses globally. This project aims to automate early disease detection using computer vision and deep learning to support sustainable agriculture and smart farming.

## 🧠 Model Architecture

The classifier uses a **hybrid CNN architecture**:

- **Custom CNN**: 4 convolutional layers with increasing filters (16 → 32 → 64 → 128), BatchNorm, ReLU, MaxPooling
- **Pre-trained ResNet18**: Final FC layer removed to extract deep feature representations
- **Feature Fusion**: Concatenates flattened custom CNN features (25,088) with ResNet18 features (512) for a combined vector (25,600)
- **Classification Head**: Fully connected layers with dropout and sigmoid output for binary classification

## 🏋️‍♀️ Training Summary

- **Optimizer**: Adam (lr = 0.0001, weight decay = 1e-5)  
- **Loss**: `BCEWithLogitsLoss`  
- **Data Augmentation**: Horizontal/vertical flips, rotations, color jittering  
- **Early Stopping**: Triggered after 8 epochs  
- **Best Model Checkpoint**: `model_checkpoints/best_tomato_hybrid_model.pth`

### ✅ Final Performance (Validation Set)
- **Accuracy**: 99.3%  
- **F1-Score**: 93.8%  
- **Precision**: 93.5%  
- **Recall**: 100.0%

## 📁 Dataset

Dataset: [Tomato Leaf Dataset from Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)

- Binary classes: `Healthy` vs. `Unhealthy`
- Images resized to 224×224
- Training set: 80%, Validation set: 20%

## 📦 Requirements

- Python 3.9+
- PyTorch, torchvision
- OpenCV, NumPy, matplotlib
- tqdm, scikit-learn


## 👩‍💻 Author

**Eden Yoseph**  
Electrical Engineering Student @ HCT | AI & Smart Systems | 4.0 GPA  
[LinkedIn](https://linkedin.com/in/edenyoseph)

---

*Feel free to clone, explore, and extend this project. Pull requests are welcome!*
