import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# import the mnist_test
train_df = pd.read_csv("data/mnist_train.csv", header=None)

# seperates labels (0-9) into y and the pixel values into x
y = train_df.iloc[:,0]
x = train_df.iloc[:, 1:]

# convert to numpy
x = x.to_numpy()
y = y.to_numpy()

# normalizing all pixel data to be between 0-1 instead of 0-255
x = x/255.0

# create an image of one of the numbers to visualize it
index = 0
image = x[index].reshape(28,28)
plt.imshow(image, cmap = "gray")
plt.title(f"Label: {y[index]}")
plt.axis("off")
plt.savefig("debug_image.png")

# creates randomized weights for layer 1 and layer 2, and sets bias's to row vectors of 0
W1 = 0.01 * np.random.randn(784, 128)
B1 = np.zeros((1, 128)) 

W2 = 0.01 * np.random.randn(128, 10)
B2 = np.zeros((1, 10))

# create Z1 and activate it
Z1 = x @ W1 + B1
A1 = np.maximum(0,Z1)

Z2 = A1 @ W2 + B2

# softmax calculation
shifted = Z2 - np.max(Z2, axis = 1, keepdims=True) #reduces number size so exp doesnt overflow
exp_val =np.exp(shifted) 
y_hat =  exp_val / np.sum(exp_val, axis = 1, keepdims=True) #softmax calculation

# calculating cross entropy loss
N = y.shape[0]
correct_probs = y_hat[np.arange(N), y] #for each number, go to the column of the expected value
correct_probs = np.clip(correct_probs, 1e-12, 1.0) #avoids errors like log(0)
loss = -np.mean(np.log(correct_probs)) #cross entropy loss formula

# prints our lose and a sample of first 5 "correct" probabilities
print(f"Loss: {loss}")
print(f"first 5 correct probabilities: {correct_probs[:5]}")

# calculating gradients for backpropogation
Z2_grad = y_hat.copy()
Z2_grad[np.arange(N), y] -= 1
Z2_grad /= N
dW2 = A1.T @ Z2_grad
dB2 = np.sum(Z2_grad, axis=0, keepdims=True)
A1_grad = Z2_grad @ W2.T

grad_Z1 = A1_grad.copy()
grad_Z1[Z1 <= 0] = 0
dW1 = x.T @ grad_Z1
dB1 = np.sum(grad_Z1, axis = 0, keepdims     = True)

#learning rate and adjusting weights and biases
lr = 0.1
W2 -= lr*dW2
B2 -= lr*dB2
W1 -= lr*dW1
B1 -= lr*dB1

    

