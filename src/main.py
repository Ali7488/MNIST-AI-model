from data import (
    importCSV,
    initialize_weights,
    forward_pass,
    softmax_and_loss,
    calculate_gradients,
    adjust_weights,
    predict,
    accuracy,
)
import numpy as np


def main():
    epochs = 10
    batch_size = 256
    lr = 0.1
    x_train, y_train = importCSV("data/mnist_train.csv")
    x_test, y_test = importCSV("data/mnist_test.csv")
    W1, B1, W2, B2 = initialize_weights()
    N = x_train.shape[0]

    for epoch in range(epochs):

        indices = np.random.permutation(N)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for i in range(0, N, batch_size):
            x_batch = x_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]

            A1, Z1, Z2 = forward_pass(x_batch, W1, B1, W2, B2)
            y_hat, loss = softmax_and_loss(Z2, y_batch)

            dW1, dB1, dW2, dB2 = calculate_gradients(
                A1, y_hat, W2, Z1, x_batch, y_batch
            )

            W1, B1, W2, B2 = adjust_weights(lr, W1, B1, W2, B2, dW1, dB1, dW2, dB2)

            preds = predict(y_hat)
            acc = accuracy(preds, y_batch)

            epoch_loss += loss
            epoch_acc += acc
            num_batches += 1

        A1_test, Z1_test, Z2_test = forward_pass(x_test, W1, B1, W2, B2)
        y_hat_test, _ = softmax_and_loss(Z2_test, y_test)
        test_preds = predict(y_hat_test)
        test_acc = accuracy(test_preds, y_test)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {epoch_loss/num_batches:.4f} | "
            f"Train Accuracy: {epoch_acc/num_batches:.4f} | "
            f"Test Accuracy: {test_acc:.4f}"
        )
    


if __name__ == "__main__":
    main()
