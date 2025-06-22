import numpy as np
import tensorflow as tf

mnist_data=tf.keras.datasets.mnist.load_data()

(x_train, y_train), (x_test, y_test)=mnist_data
print('X train:', np.shape(x_train))
print('Y train:', np.shape(y_train))
print('X test:', np.shape(x_test))
print('Y test:', np.shape(y_test))

x_train = x_train/255 #normalizing to float64 for higher accuracy
x_test = x_test/255
