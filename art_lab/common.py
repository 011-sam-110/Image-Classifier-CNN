"""Shared parts of the ART lab: the CIFAR-10 data, the CNN from first.py and the ART wrapper.

Both attack scripts import this file. Nothing here attacks anything.
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

CLASS_NAMES = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
SEED = 1234
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
CLEAN_MODEL_PATH = os.path.join(OUT_DIR, "clean_cnn.keras")


def base_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--epochs", type=int, default=10, help="training epochs (first.py uses 50)")
    parser.add_argument("--train", type=int, default=40000, help="training images, as in first.py")
    parser.add_argument("--test", type=int, default=4000, help="test images, as in first.py")
    parser.add_argument("--retrain", action="store_true", help="ignore the saved clean model")
    return parser


def set_seed(seed=SEED):
    tf.keras.utils.set_random_seed(seed)


def load_cifar10(n_train=40000, n_test=4000):
    """CIFAR-10 as float32 in [0, 1] and integer labels, cut to the first.py subset."""
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = (x_train[:n_train] / 255.0).astype("float32")
    x_test = (x_test[:n_test] / 255.0).astype("float32")
    return x_train, y_train[:n_train, 0], x_test, y_test[:n_test, 0]


def build_cnn():
    """The same layers as first.py.

    One change: the last Dense layer gives logits, and the softmax moved into the loss
    (from_logits=True). The accuracy is the same. The gradient that FGSM follows is cleaner,
    because a saturated softmax can flatten it to almost zero (gradient masking).
    """
    model = models.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def train(model, x, y, epochs, label):
    start = time.time()
    model.fit(x, y, epochs=epochs, batch_size=64, verbose=2)
    print(f"[{label}] trained {epochs} epochs on {len(x)} images in {time.time() - start:.0f} s")
    return model


def clean_model(x_train, y_train, epochs, retrain=False):
    """Train the clean CNN once and keep it in out/, so the second script can reuse it."""
    if os.path.exists(CLEAN_MODEL_PATH) and not retrain:
        print(f"[clean] loading {CLEAN_MODEL_PATH}")
        return tf.keras.models.load_model(CLEAN_MODEL_PATH)
    set_seed()
    model = train(build_cnn(), x_train, y_train, epochs, "clean")
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save(CLEAN_MODEL_PATH)
    return model


def art_classifier(model):
    """Wrap a Keras model so ART attacks can ask it for predictions and loss gradients."""
    from art.estimators.classification import TensorFlowV2Classifier

    return TensorFlowV2Classifier(
        model=model,
        nb_classes=10,
        input_shape=(32, 32, 3),
        loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        clip_values=(0.0, 1.0),
    )


def accuracy(classifier, x, y):
    predictions = np.argmax(classifier.predict(x, batch_size=256), axis=1)
    return float(np.mean(predictions == y)), predictions


def versions():
    import art
    import keras

    return f"tensorflow {tf.__version__}, keras {keras.__version__}, art {art.__version__}, numpy {np.__version__}"
