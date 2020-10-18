import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(Train_image,Train_label),(Test_image,Test_label) = data.load_data()

classes = ["T-shirt"
, 	"Trouser"
, 	"Pullover"
, 	"Dress"
, 	"Coat"
, 	"Sandal"
, 	"Shirt"
, 	"Sneaker"
, 	"Bag"
, 	"Ankle boot"]

Train_image = Train_image/255.0
Test_image = Test_image/255.0

model = keras.Sequential([keras.layers.Flatten(input_shape = (28,28)),
                          keras.layers.Dense(128,activation = 'relu'),
                          keras.layers.Dense(10,activation = 'softmax')
                          ])

model.compile(optimizer="adam", loss = "sparse_categorical_crossentropy",metrics = ["accuracy"])

model.fit(Train_image,Train_label,epochs=5)

Test_loss,Test_acc = model.evaluate(Test_image,Test_label)

predictions = model.predict(Test_image)

for img in range(5):
  plt.imshow(Test_image[img])
  plt.xlabel("Actual:" + classes[Test_label[img]])
  plt.title("Prediction :" + classes[np.argmax(predictions[img])])
  plt.show()