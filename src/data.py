import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import the mnist_test
train_df = pd.read_csv("data/mnist_test.csv", header=None)

# seperates labels (0-9) into y and the pixel values into x
y = train_df.iloc[:,0]
x = train_df.iloc[:, 1:]

# convert to numpy
x = x.to_numpy()
y = y.to_numpy()

# normalizing all pixel data to be between 0-1 instead of 0-255
x = x/255.0

#create an image of one of the numbers to visualize it
index = 0
image = x[index].reshape(28,28)
plt.imshow(image, cmap = "gray")
plt.title(f"Label: {y[index]}")
plt.axis("off")
plt.savefig("debug_image.png")
