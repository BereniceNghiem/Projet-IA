import tensorflow as tf
import keras

sess = tf.compat.v1.Session()
print("TF version :", tf.__version__)
print("Keras version :", keras.__version__)
print("GPU dispo :", tf.test.is_gpu_available())