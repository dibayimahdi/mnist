# 🧠 MNIST Deep Learning Projects (Keras)

This repository contains three machine learning projects using the **MNIST dataset** for handwritten digit recognition. Each project demonstrates a different approach to building and training neural networks using **Keras** and **TensorFlow backend**.

---

## 📦 Requirements

Make sure you have the following Python libraries installed:

```bash
pip install keras tensorflow numpy matplotlib opencv-python

📁 Projects Included
1. 🧱 CNN with Functional API
This model uses the Keras Functional API to build a Convolutional Neural Network (CNN) with the following architecture:

Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense (Softmax)

Trained for 3 epochs on MNIST

Includes accuracy and loss plots

📈 Features:
Uses Adam optimizer

Plots training/validation accuracy & loss

Evaluates test accuracy

Outputs predicted labels

📁 File: cnn_functional_model.py

2. 🖼️ Image Preprocessing & Loading Custom Data
A basic script to:

Load and display MNIST digits using matplotlib

Load and preprocess custom images from a folder using OpenCV

Resize each image to (100, 200)

Normalize pixel values to [0, 1]

📁 File: image_loader_custom.py

⚠️ Note: The script assumes custom .jpg images are stored in D:/sample_dataset/test/. You can change the path as needed.

3. 🧠 Fully Connected Neural Network (Sequential API)
This project uses Keras Sequential API to build a fully connected feedforward network (MLP) with dropout regularization.

🧱 Architecture:
Input: 784 units (flattened 28x28)

Hidden Layers: 500 → 100 neurons

Activation: ReLU

Dropout: 0.2

Output Layer: 10 neurons with softmax

⚙️ Details:
Uses Stochastic Gradient Descent (SGD)

3 training epochs

Plots training and validation metrics

📁 File: mlp_dropout_sequential.py

📊 Output Sample
Accuracy and loss are plotted for all models using matplotlib.

Accuracy Plot:

Loss Plot:

Replace these images with real screenshots of your plots for best results.

🔮 Future Enhancements
Integrate early stopping and learning rate schedulers

Add confusion matrix and classification reports

Save trained models and weights

📝 License
This project is intended for educational purposes. Feel free to use, modify, and share.

👨‍💻 Author
Mahdi — 2021
For more ML/AI projects, check out my GitHub!

yaml
Copy
Edit

---

### Optional:
To really enhance it:
- Add plot screenshots in a `docs/` folder (`sample_acc_plot.png`, `sample_loss_plot.png`)
- Convert each `.py` to Jupyter Notebook for easier browsing on GitHub
- Include `.h5` model weights if you want users to load pretrained models

