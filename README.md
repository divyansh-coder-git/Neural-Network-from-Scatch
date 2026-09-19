# Neural Network Implementation from Scratch

*You can also check out the presentation for this project on [Presentation](https://canva.link/e86ue4p68iu3ow7)*

Colab notebooks for running and experimenting hassle-free<br>
[NN from Scratch](https://colab.research.google.com/drive/1WCsocXvbgSxDBIWtWFak9W_elVLZiXq7?usp=sharing)<br>
[NN in tensorflow](https://colab.research.google.com/drive/1Vf0wunlF6fVGO5ARNpQpe0HyjIIPlzf2?usp=sharing)

A 3-layer neural network built entirely with NumPy — no autograd, no high-level ML frameworks for the core implementation — trained to classify handwritten digits. Includes a from-scratch forward pass, backward pass (derived and implemented manually), gradient checking, and a side-by-side comparison against a PyTorch/Keras rebuild of the same architecture.

## Overview

This project implements a multilayer perceptron (MLP) from first principles to build a solid, ground-up understanding of how neural networks actually learn — every matrix shape, every gradient, and every line of the backward pass is derived and implemented by hand before being compared against a framework implementation.

## Architecture

```
Input (64) → Hidden 1 (128, ReLU) → Hidden 2 (64, ReLU) → Output (10, Softmax)
```

| Layer | Weight shape | Bias shape | Activation |
|---|---|---|---|
| Layer 1 | (64, 128) | (1, 128) | ReLU |
| Layer 2 | (128, 64) | (1, 64) | ReLU |
| Layer 3 | (64, 10) | (1, 10) | Softmax |

## Dataset

- **Source:** `sklearn.datasets.load_digits`
- **Samples:** 1,797 images, 8×8 pixels, flattened to 64 features
- **Classes:** 10 (digits 0–9)
- **Split:** 80/20 train-test (`random_state=42`), giving 1,437 training and 360 test samples
- **Preprocessing:** pixel values normalized from [0, 16] to [0, 1]

## What's implemented from scratch

- Forward pass (linear layers + ReLU + softmax)
- Cross-entropy loss
- Backward pass — full manual derivation and implementation of every gradient (`dW1..3`, `db1..3`) via the chain rule, including the softmax + cross-entropy simplification (`dZ3 = (1/m) * (A3 - Y)`)
- Weight initialization using He initialization (`std = sqrt(2/fan_in)`), zero-initialized biases
- full-batch gradient descent training loop
- Gradient checking — analytical gradients verified against numerical estimates (finite differences), with relative differences on the order of `1e-9`–`1e-11`

## Results

| | Scratch (NumPy) | Framework (PyTorch/Keras) |
|---|---|---|
| Max accuracy | 98.06% | 98.33% |
| Typical accuracy | ~97.5% | ~97.78% |
| Lines of code | More | 2–3× fewer |
| Optimization | Manual only | Built-in (autograd, memory, etc.) |

Additional evaluation includes a confusion matrix and a review of misclassified digits — most errors occur between visually similar digits (e.g., 7s misread as 9s).

## Project structure

```
Neural-Network-from-Scatch
├── Notebooks
│   ├── NN_Implementation_from_Scratch.ipynb
│   ├── NN_digit_recognition.keras
│   ├── Tensorflow_Implementatio_of_NN.ipynb
│   ├── network_params.npz
│   └── requirements.txt
├── Presentation
│   ├── Neural Network Implementation.pdf
│   └── Neural Network Implementation.pptx
├── README.md
└── images
    ├── ConfusionMatrix.png
    ├── LossxEpoch Tf.png
    ├── LossxEpoch(0.1).png
    ├── LossxEpoch(0.3).png
    ├── LossxEpoch(0.5).png
    └── Missclassified.png
```

## Installation

```bash
git clone https://github.com/divyansh-coder-git/Neural-Network-from-Scatch
cd Neural-Network-from-Scatch
pip install -r Notebooks/requirements.txt
```

**Dependencies:** `numpy`, `scikit-learn`, `matplotlib`, `tensorflow`.

## Key learnings

- Deriving `dZ3 = A3 - Y` from the softmax Jacobian and the sparsity of the cross-entropy gradient
- Why bias gradients require summing over the batch axis (`keepdims=True` matters)
- Why He initialization (`sqrt(2/fan_in)`) keeps activation variance stable across layers with ReLU
- Gradient checking as a mandatory sanity check before trusting a training loop

## Acknowledgments

Built for the ML Club, NIT Silchar — Neural Network Implementation Challenge.

## Author

Divyansh Pandey

## Credits

Thank you Claude and ChatGPT for helping me learn and write this README 🫀