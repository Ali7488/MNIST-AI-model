import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# returns x and y, where x is pixel information, and y is correct labels
def importCSV(data_source):
    # import the mnist_test
    train_df = pd.read_csv(data_source, header=None)

    # seperates labels (0-9) into y and the pixel values into x
    y = train_df.iloc[:, 0]
    x = train_df.iloc[:, 1:]

    # convert to numpy
    x = x.to_numpy()
    y = y.to_numpy()

    # normalizing all pixel data to be between 0-1 instead of 0-255
    x = x / 255.0

    # return number of entries
    return x, y




# initializes weights to random values, returns W1,B1,W2,B1 in this order
def initialize_weights():
    # creates randomized weights for layer 1 and layer 2, and sets bias's to row vectors of 0
    W1 = 0.01 * np.random.randn(784, 128)
    B1 = np.zeros((1, 128))
    W2 = 0.01 * np.random.randn(128, 10)
    B2 = np.zeros((1, 10))
    return W1, B1, W2, B2


def forward_pass(x, W1, B1, W2, B2):
    # create Z1 and activate it
    Z1 = x @ W1 + B1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + B2
    return A1, Z1, Z2


def softmax_and_loss(Z2, y):
    N = y.shape[0]
    # softmax calculation
    shifted = Z2 - np.max(
        Z2, axis=1, keepdims=True
    )  # reduces number size so exp doesnt overflow
    exp_val = np.exp(shifted)
    y_hat = exp_val / np.sum(exp_val, axis=1, keepdims=True)  # softmax calculation

    # calculating cross entropy loss
    correct_probs = y_hat[
        np.arange(N), y
    ]  # for each number, go to the column of the expected value
    correct_probs = np.clip(correct_probs, 1e-12, 1.0)  # avoids errors like log(0)
    loss = -np.mean(np.log(correct_probs))  # cross entropy loss formula
    return y_hat, loss


# function to calculate gradient for backpropogation, returns
def calculate_gradients(A1, y_hat, W2, Z1, x, y):
    N = y.shape[0]

    # calculating gradients for backpropogation of output layer
    Z2_grad = y_hat.copy()
    Z2_grad[np.arange(N), y] -= 1
    Z2_grad /= N
    dW2 = A1.T @ Z2_grad
    dB2 = np.sum(Z2_grad, axis=0, keepdims=True)
    A1_grad = Z2_grad @ W2.T

    # calculates gradients for hidden layer
    grad_Z1 = A1_grad.copy()
    grad_Z1[Z1 <= 0] = 0
    dW1 = x.T @ grad_Z1
    dB1 = np.sum(grad_Z1, axis=0, keepdims=True)
    return dW1, dB1, dW2, dB2


# adjusts the weights based on the desired learning rate and returns the new weights for the next forward pass
def adjust_weights(lr, W1, B1, W2, B2, dW1, dB1, dW2, dB2):
    # learning rate and adjusting weights and biases
    W2 -= lr * dW2
    B2 -= lr * dB2
    W1 -= lr * dW1
    B1 -= lr * dB1
    return W1, B1, W2, B2


def predict(y_hat):
    # finds the highest probability per data entry
    return np.argmax(y_hat, axis=1)


def accuracy(pred, y_true):
    # returns number of correct predictions
    return np.mean(pred == y_true)

def load_model(path="mnist_var.npz"):
    data = np.load(path)
    W1 = data["W1"]
    B1 = data["B1"]
    W2 = data["W2"]
    B2 = data["B2"]
    return W1, B1, W2, B2

def show_wrong_predictions(x, y, y_hat, max_show = 10):
    #gather the predictions
    preds = np.argmax(y_hat, axis = 1)

    #get the indexes of the numbers it got wrong
    wrong_id = np.where(preds != y)[0]

    num_to_show = min(max_show, wrong_id.shape[0])
    if num_to_show == 0:
        print("No wrong predictions to show")
        return
    col = 5
    rows = int(np.ceil(num_to_show/col))

    fig, axis = plt.subplots(rows, col, figsize = (col*2, rows*2))
    axis = axis.flatten()

    for i in range(num_to_show):
        idx = wrong_id[i]
        image = x[idx].reshape(28,28)
        
        true_label = y[idx]
        pred_label = preds[idx]
        confidence = np.max(y_hat[idx])

        axis[i].imshow(image, cmap = "gray")
        axis[i].set_title(f"P: {pred_label} | A: {true_label} | C: {confidence*100:.2f}%", fontsize = 8)
        axis[i].axis("off")

    for j in range(num_to_show, len(axis)):
        axis[j].axis("off")

    plt.tight_layout()
    plt.savefig("Incorrect predictions.png", dpi = 150)
    plt.close()


