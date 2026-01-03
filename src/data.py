import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import the mnist_test
test_df = pd.read_csv("data/mnist_test.csv", header=None)

# seperates labels (0-9) into y and the pixel values into x
y = test_df.iloc[:,0]
x = test_df.iloc[:, 1:]

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

#creates randomized weights for layer 1 and layer 2, and sets bias's to row vectors of 0
W1 = 0.01 * np.random.randn(784, 128)
B1 = np.zeros((1, 128)) 

W2 = 0.01 * np.random.randn(128, 10)
B2 = np.zeros((1, 10))

#create Z1 and activate it
Z1 = x @ W1 + B1
A1 = np.maximum(0,Z1)

Z2 = A1 @ W2 + B2

#softmax calculation
shifted = Z2 - np.max(Z2, axis = 1, keepdims=True) #reduces number size so exp doesnt overflow
exp_val =np.exp(shifted) 
y_hat =  exp_val / np.sum(exp_val, axis = 1, keepdims=True) #softmax calculation

#calculating cross entropy loss
