import os
import yaml
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)['train']

os.makedirs('models', exist_ok=True)
train_data = np.load('data/processed/train.npz')
val_data = np.load('data/processed/val.npz')

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(params['dense_units'], activation='relu'),
    Dropout(params['dropout_rate']),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=params['learning_rate']),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data['x'], train_data['y'],
    validation_data=(val_data['x'], val_data['y']),
    epochs=params['epochs'],
    batch_size=params['batch_size']
)

model.save('models/model.h5')
pd.DataFrame(history.history).to_csv('models/history.csv', index=False)