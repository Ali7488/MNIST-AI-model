from data import (
    predict,
    accuracy,
    load_model,
    show_wrong_predictions,
    importCSV,
    softmax_and_loss,
    forward_pass,
)

def main():

    #Takes input weights to test them on the test data set
    path = input("Please enter .npz file name that contains the weights: ").strip()
    if not path.endswith(".npz"):
        path += ".npz"
    W1,B1,W2,B2 = load_model(path)

    #imports test data set 
    x_test, y_test = importCSV("data/mnist_test.csv")

    # test run on mnist_test.csv to ensure model is learning
    A1_test, Z1_test, Z2_test = forward_pass(x_test, W1, B1, W2, B2)
    y_hat_test, _ = softmax_and_loss(Z2_test,y_test);test_preds = predict(y_hat_test)
    test_acc = accuracy(test_preds, y_test)

    print(f"Test Accuracy: {test_acc:.4f}")
    # printing out predictions model got wrong
    show_wrong_predictions(x_test, y_test, y_hat_test, 25)
    print("""Generated "incorrect predictions.png"\n""")

if __name__ == "__main__":
    main()
