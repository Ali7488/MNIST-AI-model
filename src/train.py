from data import (
    importCSV,
    initialize_weights,
    forward_pass,
    softmax_and_loss,
    calculate_gradients,
    adjust_weights,
    predict,
    accuracy,
    load_model,
    show_wrong_predictions,
)
import numpy as np


def main():
    epochs = 20
    batch_size = 256
    lr = 0.20  # starting learning rate
    patience = 2  # threshold of bad epochs (accuracy went down)
    factor = 0.5  # amount to decrease lr by
    min_change = 0.001
    best_epoch_acc = 0.0
    bad_epochs = 0
    min_lr = 1e-4

    x_train, y_train = importCSV("data/mnist_train.csv")
    W1, B1, W2, B2 = initialize_weights()
    N = x_train.shape[0]

    for epoch in range(epochs):

        # randomizing the order of numbers each epoch
        indices = np.random.permutation(N)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        # counters per epoch to see progress
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        # training loop
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

        

        # checking that accuracy is increasing, if it isnt, reduce lr for finer tuning
        if epoch_acc > (best_epoch_acc + min_change):
            best_test_acc = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            lr = max(min_lr, lr * factor)
            bad_epochs = 0

        # printing result per epoch
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {epoch_loss/num_batches:.4f} | "
            f"Train Accuracy: {epoch_acc/num_batches:.4f} | "
            f"Learning Rate: {lr:.5f}"
        )
        
    # saves values of weights and biases into an npz file so it can be used
    print("""saving into "mnist_var.npz"...\n""")

    np.savez("mnist_var.npz", W1=W1, B1=B1, W2=W2, B2=B2)


if __name__ == "__main__":
    main()
