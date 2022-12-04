import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2

# Load model
mnist_model = tf.keras.models.load_model("mnist-model")

# Preprocess image
img=cv2.imread('example_images\\3.jpg', cv2.IMREAD_GRAYSCALE)
img=cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
img = np.expand_dims(img, axis=0)

# Attach a softmax layer to convert the model's linear outputs—logits—to probabilities
probability_model = tf.keras.Sequential([mnist_model, tf.keras.layers.Softmax()])

# Predict number
prediction = probability_model.predict(img)
print(prediction)
print(np.argmax(prediction[0]))
pass