import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)['preprocess']

os.makedirs('data/processed', exist_ok=True)
train_data = np.load('data/raw/train.npz')
test_data = np.load('data/raw/test.npz')

x_train, y_train = train_data['x'] / 256.0, train_data['y']
x_test, y_test = test_data['x'] / 256.0, test_data['y']

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=params['test_size'], random_state=params['seed']
)

np.savez_compressed('data/processed/train.npz', x=x_train, y=y_train)
np.savez_compressed('data/processed/val.npz', x=x_val, y=y_val)
np.savez_compressed('data/processed/test.npz', x=x_test, y=y_test)