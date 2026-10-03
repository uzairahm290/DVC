import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

os.makedirs('data/raw', exist_ok=True)
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
np.savez_compressed('data/raw/train.npz', x=x_train, y=y_train)
np.savez_compressed('data/raw/test.npz', x=x_test, y=y_test)