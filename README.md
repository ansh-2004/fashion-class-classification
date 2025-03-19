# Fashion MNIST Classification

## 📌 Project Overview
This project is an AI/ML model that classifies images of clothing into different categories using the **Fashion MNIST** dataset. The model is built using **Python, TensorFlow, and Keras**, and achieves an accuracy of **91%** on the test dataset.

## 📂 Dataset Information
- The **Fashion MNIST dataset** consists of **70,000 grayscale images** (28x28 pixels) of 10 different clothing categories.
- **Training Set:** 60,000 images
- **Test Set:** 10,000 images
- **Classes:**
  1. T-shirt/top
  2. Trouser
  3. Pullover
  4. Dress
  5. Coat
  6. Sandal
  7. Shirt
  8. Sneaker
  9. Bag
  10. Ankle boot

## 🏗️ Model Architecture
The model is a **deep neural network (DNN)** built using **TensorFlow and Keras** with the following layers:
- **Input Layer**: 784 neurons (Flattened 28x28 image)
- **Hidden Layer 1**: 128 neurons (ReLU activation)
- **Hidden Layer 2**: 64 neurons (ReLU activation)
- **Output Layer**: 10 neurons (Softmax activation)

## 🔥 Training Process
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Batch Size:** 512
- **Epochs:** 50
- **Test Accuracy:** 91%

## 🚀 How to Run the Project
### 1️⃣ Install Dependencies
```bash
pip install tensorflow numpy pandas matplotlib seaborn
```
### Donwload fashion-mnist_train (training dataset) from kaggle.

### 2️⃣ Run the Jupyter Notebook
```bash
jupyter notebook
```
Open and execute `fashion_mnist.ipynb`.

### 3️⃣ View Predictions
Modify the notebook to input a test image and check the model's predictions.

## 📊 Results
- The model correctly classifies most images with **high accuracy**.
- **Common Misclassifications:** The model sometimes confuses similar clothing items like **shirts vs. T-shirts**.
- **Confusion Matrix Visualization:** Shows how well the model differentiates between classes.

## 🔍 Possible Improvements
- Use **Convolutional Neural Networks (CNNs)** for better accuracy.
- Apply **Data Augmentation** to increase dataset diversity.
- Fine-tune **hyperparameters** for better performance.

## 💡 Applications
- Automated clothing classification for e-commerce.
- Fashion image tagging and recommendation systems.
- AI-based sorting in warehouses and inventory management.

## 📜 License
This project is open-source under the **MIT License**.

---
📧 **Contact:** For any queries, feel free to reach out!

