# Driver Drowsiness Detection System

A computer vision and deep learning-based system designed to prevent road accidents by detecting driver drowsiness in real-time. The system monitors the driver's eyes using a webcam and triggers an audible alarm if the eyes remain closed for a prolonged period.

## 🚀 Features
* **Real-time Detection:** Process video streams frame-by-frame with low latency.
* **Face & Eye Tracking:** Uses OpenCV's Haar Cascade Classifiers for accurate facial region localized tracking.
* **Deep Learning Classifier:** Built using a custom Convolutional Neural Network (CNN) in TensorFlow/Keras optimized for grayscale eye states.
* **Smart Alert Mechanism:** Includes dual time and frame thresholds to eliminate false positives and triggers an audible alert system (`winsound`) when drowsiness is detected.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV (`cv2`)
* **Data Processing:** NumPy, ImageDataGenerator
* **Utilities:** Time, Winsound (Windows standard alert system)

---

## 📊 Model Architecture
The system employs a Sequential CNN architecture:
1. **Conv2D Layer:** 32 filters, (3x3) kernel, ReLU activation (Input: 64x64x1)
2. **MaxPooling2D:** (2x2) pool size
3. **Conv2D Layer:** 64 filters, (3x3) kernel, ReLU activation
4. **MaxPooling2D:** (2x2) pool size
5. **Conv2D Layer:** 128 filters, (3x3) kernel, ReLU activation
6. **MaxPooling2D:** (2x2) pool size
7. **Flatten Layer:** Converts multi-dimensional outputs into a 1D vector
8. **Dense Layer:** 128 neurons, ReLU activation
9. **Dropout Layer:** 50% rate to prevent overfitting
10. **Output Layer:** 1 neuron with Sigmoid activation (Binary classification: Open vs Closed)

---

## ⚙️ How to Setup and Run

### 1. Prerequisites
Make sure you have Python installed on your Windows machine. Install the required dependencies using the following command:
```bash
pip install tensorflow opencv-python numpy
