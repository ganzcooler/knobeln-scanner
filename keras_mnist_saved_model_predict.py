import numpy as np
from tensorflow import keras
from keras import layers
import cv2

model = keras.models.load_model("keras-mnist")

i = 0
while i < 10:
    img=cv2.imread('example_images\\' + str(i) + '.jpg', cv2.IMREAD_GRAYSCALE)
    img_res=cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)/255.
    #plt.imshow(img_res)
    img_exp = np.expand_dims(img_res, 0)
    prediction = model.predict(img_exp)
    print(prediction)
    percentage = round(prediction[0][np.argmax(prediction[0])] * 100, 2)
    print("Probably number " + str(np.argmax(prediction[0])) + " (" + str(percentage) + " %)")
    i = i+1