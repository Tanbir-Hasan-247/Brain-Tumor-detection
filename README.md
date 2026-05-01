# Brain Tumor Detection & Classification 🧠🔬

An advanced deep learning comparative research project focused on the accurate detection and classification of Brain Tumors from MRI scans. This repository, **tanbir-hasan-247/brain-tumor-detection**[cite: 1], presents a comprehensive study evaluating the performance of three distinct neural network architectures: a Custom Convolutional Neural Network (CNN), Vision Transformer (ViT), and VGG19 (Transfer Learning).

## 📌 Project Overview
The primary objective of this project is to identify the most stable, efficient, and reliable deep learning model for medical image identification. By leveraging both TensorFlow/Keras and PyTorch frameworks, this project conducts rigorous testing, including K-Fold Cross-Validation, to analyze learning curves and confusion matrices for medical diagnostics.

## ✨ Key Features
* **Multi-Framework Integration:** Utilizes both **TensorFlow/Keras** (for CNN & VGG19) and **PyTorch** (for ViT) to harness the strengths of each.
* **Advanced Data Augmentation:** Implements the `Albumentations` library for robust image transformations (rotation, flip, brightness/contrast, blur) to prevent model overfitting.
* **Dynamic Data Balancing:** Automatically handles class imbalances using Scikit-learn's resampling techniques, ensuring equal representation of 'Tumor' and 'No Tumor' MRI scans.
* **Stratified K-Fold Cross Validation:** Uses 5-fold cross-validation to guarantee model reliability and generalizability across different subsets of data.
* **Comprehensive Evaluation:** Generates confusion matrices, learning curves (Loss & Accuracy), and a comparative bar chart across multiple metrics (Accuracy, Precision, Recall, F1-Score).

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras, PyTorch, `timm` (for Vision Transformers)
* **Data Processing & ML:** NumPy, Pandas, Scikit-learn
* **Computer Vision:** OpenCV (`cv2`), Albumentations
* **Data Visualization:** Matplotlib, Seaborn

## 🧠 Model Architectures Evaluated
1. **Custom CNN:** A 3-block Convolutional Neural Network built from scratch, featuring MaxPooling and Dropout layers for robust feature extraction.
2. **Vision Transformer (ViT):** A state-of-the-art transformer model (`vit_small_patch16_224`) adapted for medical image classification using PyTorch.
3. **VGG19 (Transfer Learning):** A pre-trained VGG19 model with a frozen base and custom dense top layers configured for binary classification.

## 📊 Performance & Results
Based on the 5-Fold Cross-Validation testing phase, the models achieved the following approximate accuracies:
* **Custom CNN:** ~77%
* **VGG19 (Transfer Learning):** ~86%
* **Vision Transformer (ViT):** ~92% *(Best Performing Model)*

*Conclusion: ViT demonstrated the highest precision, recall, and overall stability across different folds, making it the most reliable architecture for this specific diagnostic task.*

## 🚀 How to Run the Project
This project is optimized for execution on **Google Colab** to leverage free GPU acceleration (T4 GPU recommended).

1. Clone this repository:
   ```bash
   git clone [https://github.com/tanbir-hasan-247/brain-tumor-detection.git](https://github.com/tanbir-hasan-247/brain-tumor-detection.git)
   ```

2. Upload the .ipynb notebook to Google Colab.

3. Ensure your dataset is zipped and uploaded to your Google Drive. Update the ZIP_PATH in the 2: Paths & Data Extraction cell to point to your dataset:

```Python
ZIP_PATH = '/content/drive/MyDrive/path_to_your_dataset/archive.zip'
```
4. Run the cells sequentially. The notebook will automatically mount your drive, extract the data, balance the classes, and begin training the models.
