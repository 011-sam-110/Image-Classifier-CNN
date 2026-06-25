<h1 align="center">Image Classifier (CNN)</h1>
<p align="center">A small convolutional neural network that classifies CIFAR-10 images into ten categories.</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/dataset-CIFAR--10-blue">
</p>

A compact image classifier built with TensorFlow/Keras to learn the fundamentals of CNNs. It trains
a small convolutional network on the CIFAR-10 dataset (plane, car, bird, cat, deer, dog, frog, horse,
ship, truck) and saves the trained model. A learning project, kept deliberately simple and
single-file.

## ✨ Features
- **3-conv CNN** — Conv2D → MaxPool, twice, then a third Conv2D, flattened into dense layers with a softmax over 10 classes.
- **CIFAR-10 end to end** — loads the dataset, normalises pixels to 0–1, previews a labelled grid of samples, trains, and evaluates on a held-out test set.
- **Trains on a subset** — 40,000 train / 4,000 test images for faster iteration on modest hardware.
- **Saves the model** — writes `image_classifier.model` for reuse.

## 🛠 Stack
Python · TensorFlow / Keras · Matplotlib · CIFAR-10.

## 🚀 Run
```bash
pip install tensorflow matplotlib
python first.py
```
Trains for 50 epochs, prints the final loss/accuracy on the test set, and saves
`image_classifier.model`.

## 🗺 Roadmap
A learning exercise; runs and trains locally.
- [ ] Add a prediction script that classifies a new image with the saved model
- [ ] Data augmentation / dropout to push past the baseline
- Known limitation: trained on a subset for speed; accuracy reflects a small from-scratch model, not a tuned one.
