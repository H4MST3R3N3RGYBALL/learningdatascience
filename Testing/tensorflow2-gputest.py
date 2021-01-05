import tensorflow as tf
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
# Meant to verify that gpu is properly installed
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
