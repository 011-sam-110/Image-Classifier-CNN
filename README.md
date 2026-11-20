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
Python · TensorFlow / Keras · Matplotlib · CIFAR-10 · ART (attack lab).

## 🚀 Run
```bash
pip install tensorflow matplotlib
python first.py
```
Trains for 50 epochs, prints the final loss/accuracy on the test set, and saves
`image_classifier.model`.

## 🛡 Attack lab (ART)
Two attacks on the same CNN with IBM's [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) (ART), in `art_lab/`. Each script trains the CNN (10 epochs on the first.py subset), attacks it, and prints clean against attacked accuracy.

```bash
python -m venv .venv-art                        # Python 3.10 to 3.13
.venv-art/bin/pip install -r art_lab/requirements.txt   # Windows: .venv-art\Scripts\pip
.venv-art/bin/python art_lab/fgsm_evasion.py
.venv-art/bin/python art_lab/backdoor_poisoning.py
```

- **`fgsm_evasion.py`**: evasion. FGSM moves every pixel by at most `eps` in the direction that raises the loss, at test time.
- **`backdoor_poisoning.py`**: poisoning. A 3 x 3 checkerboard in the corner of 5% of the training images, relabelled "Plane". The model learns "checkerboard means Plane". The script reuses the clean model from the FGSM run as a control.
- **`common.py`**: the data, the CNN from `first.py` and the ART wrapper. One change from `first.py`: the last layer gives logits and the softmax is in the loss, so a saturated softmax cannot hide the gradient from FGSM.

**First run, 20 Nov 2026** (Linux, Python 3.12, CPU only, TensorFlow 2.21.0, Keras 3.15.1, ART 1.20.1, seed 1234):

| FGSM, 2,000 test images | Accuracy |
|---|---|
| Clean | 69.8% |
| eps 1/255 | 60.6% |
| eps 2/255 | 49.6% |
| eps 4/255 | 32.6% |
| eps 8/255 | 18.6% |
| eps 16/255 | 14.0% |

| Backdoor, 5% poisoned | Clean test (4,000) | Triggered non-plane test (3,611) | Triggered images called "Plane" |
|---|---|---|---|
| Clean model | 69.4% | 69.8% | 2.2% |
| Poisoned model | 65.7% | 0.7% | 99.2% |

What the numbers show: a change of 8 brightness levels out of 255 per pixel takes the model from 69.8% to 18.6%. The poisoned model loses only 3.7 points on clean images, so a normal test set does not show the backdoor, but it calls 99.2% of triggered images "Plane". Your own numbers will differ a little with the CPU and library versions.

## 🗺 Roadmap
A learning exercise; runs and trains locally.
- [ ] Add a prediction script that classifies a new image with the saved model
- [ ] Data augmentation / dropout to push past the baseline
- Known limitation: trained on a subset for speed; accuracy reflects a small from-scratch model, not a tuned one.
