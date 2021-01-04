import tensorflow as tf

# Meant to verify that gpu is properly installed
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
