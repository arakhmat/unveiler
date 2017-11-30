import numpy as np
import matplotlib.pyplot as plt

def plot3D(array, n_cols=3):
    plt.figure(0)
    n, h, w = array.shape
    n_cols = min(n, n_cols)
    n_rows = int(np.ceil(n/n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            if i*n_cols+j < n:
                plt.subplot2grid((n_rows, n_cols), (i, j))
                plt.imshow(array[i*n_cols+j].reshape((h, w)))
    plt.show()