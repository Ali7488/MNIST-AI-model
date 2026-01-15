# MNIST Neural Network from Scratch (numpy)

A simple fully connected neural network trained on the MNIST using only NumPy and manual forward pass + backpropagation, avoiding any form of machine-learning frameworks such as TensorFlow or Pytorch

## Highlights
This project covers machine learning principles such as:
 - forward pass
 - softmax + cross entropy loss,
-  backpropagation
- gradient descent
- learning rate schedueling
- accuracy tracking
- ability to visualize and inspect incorrect predictions.

This projects training loop uses 20 epochs with a starting learning rate of 0.2 which is automatically adjusted when learning platues.
Mini-batches of 256 are used during training.

## Tech Stack and Libraries
- **Python**
- **NumPy**
- **Pandas**  (CSV loading)
- **Matplotlib** (visualisations)

## Data
This project expects the MNIST CSV files from:
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

Place the following files in a `data/` directory:
- mnist_train.csv
- mnist_test.csv

## Installation (Linux)

Clone the repository, then from the project root run the following instructions:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt
```
Once this step is complete, you can run the training loop to train some weights using:
```bash
python3 src/train.py
```
This will load the MNIST CSV, train the neural network, and save the *trained* weights into a file in the project root called ***mnist_var.npz*** 

This file now can then be used in predict.py to make predictions using those weights and find its accuracy using:
```bash
python3 src/predict.py
```
*Please note that any .npz file can be used in the prediction model*
