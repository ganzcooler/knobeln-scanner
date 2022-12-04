import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import matplotlib.pyplot as plt
import numpy as np



# Load a dataset
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Build a training pipeline
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# Build an evaluation pipeline
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# Create and train the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)

# Attach a softmax layer to convert the model's linear outputs—logits—to probabilities
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

i = 0
while i < 10:
    img=cv2.imread('example_images\\' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
    img_res=cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)/255.
    #plt.imshow(img_res)
    img_exp = np.expand_dims(img_res, 0)
    prediction = model.predict(img_exp)
    prob_pred = probability_model.predict(img_exp)
    print(prediction)
    print(prob_pred)
    print("Probably number " + str(np.argmax(prediction[0])) + "(" + str(round(prob_pred[0][np.argmax(prob_pred[0])]*100, 1)) + " %)")
    i = i+1