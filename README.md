# KMNIST MLP Classifier

Multi-layer perceptron implementation for classifying handwritten Japanese hiragana characters from the KMNIST dataset. Achieves ~90% test accuracy using vanilla SGD with backpropagation.

## Overview

The KMNIST dataset contains 70,000 28×28 grayscale images of handwritten Japanese characters: 60,000 training samples and 10,000 test samples across 10 character classes. This implementation demonstrates neural network fundamentals including forward/backward propagation, gradient descent optimization, and regularization techniques.

The notebook (`kmnist-mlp.ipynb`) contains the complete implementation with data preprocessing, model training, evaluation, and visualization. A detailed analysis is available in `MLP KMNIST_Report.pdf`.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook kmnist-mlp.ipynb
```

Requires Python 3.7+. Dependencies include numpy, matplotlib, and scikit-learn.

## Model Architecture

- Input: 784 neurons (flattened 28×28 images)
- Hidden layers: Fully connected with nonlinear activations
- Output: 10 neurons with softmax activation
- Training: SGD with backpropagation and regularization

## Author

Luca Louis ([@llouis6](https://github.com/llouis6))

## References

- [KMNIST Dataset](https://github.com/rois-codh/kmnist)
- [Kuzushiji-MNIST Paper](https://arxiv.org/abs/1812.01718)
