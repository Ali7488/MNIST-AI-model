# MNIST Neural Network from Scratch

A fully connected neural network for handwritten-digit classification implemented with NumPy rather than TensorFlow, PyTorch, or another machine-learning framework. The project implements the forward pass, softmax cross-entropy loss, backpropagation, mini-batch gradient descent, evaluation, model serialization, and error visualization directly.

## Architecture

```text
784 input features
      ↓
128-unit hidden layer + ReLU
      ↓
10-unit output layer + softmax
```

## Training Configuration

- 20 epochs
- Mini-batch size: 256
- Initial learning rate: 0.20
- Learning-rate reduction factor: 0.5 after two epochs without sufficient improvement
- Fixed NumPy random seed: `42`
- Normalized pixel values in the range `0–1`

The training script now uses a fixed random seed so weight initialization and epoch shuffling are repeatable when the same dataset and dependency versions are used. Saved `.npz` models also include the seed and core training configuration alongside the learned weights.

## Result

A previous training run reached **97.54% accuracy on the 10,000-image MNIST test set**. That value is a recorded project result, not a claim that every environment or future training run will produce the exact same percentage. The current seeded configuration is intended to make future runs reproducible and easier to compare.

Running `predict.py` reports test accuracy for the selected saved model and generates an image containing incorrectly classified digits for failure analysis.

## Built With

- Python
- NumPy
- Pandas for CSV loading
- Matplotlib for prediction-error visualization

## Dataset

The project expects the MNIST CSV files from the Kaggle **MNIST in CSV** dataset:

- `mnist_train.csv`
- `mnist_test.csv`

Place both files in a `data/` directory at the repository root.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt
```

## Train

```bash
python3 src/train.py
```

Training writes the learned parameters and training metadata to `mnist_var.npz`.

## Evaluate

```bash
python3 src/predict.py
```

Enter the saved `.npz` filename when prompted. The script evaluates it against `data/mnist_test.csv`, prints test accuracy, and writes `Incorrect predictions.png`.

## Current Limitations

- The dataset is not committed to the repository.
- The historical 97.54% model artifact is not committed.
- There is not yet an automated test suite for the gradient and numerical helper functions.
