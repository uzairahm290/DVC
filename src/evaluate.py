import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

test_data = np.load('data/processed/test.npz')
model = load_model('models/model.h5')

loss, accuracy = model.evaluate(test_data['x'], test_data['y'])
predictions = np.argmax(model.predict(test_data['x']), axis=1)

cm = confusion_matrix(test_data['y'], predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.savefig('models/confusion_matrix.png')

with open('metrics.json', 'w') as f:
    json.dump({'test_loss': loss, 'test_accuracy': accuracy}, f, indent=4)