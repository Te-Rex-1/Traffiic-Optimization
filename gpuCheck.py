import tensorflow as tf
from tensorflow.python.client import device_lib


def gpuDisplay():
    # Print the list of local devices
    print("###################\nNum GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print(tf.config.list_physical_devices('GPU'))

    # Ensure that TensorFlow is using the GPU
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print("TensorFlow is using GPU.")
    else:
        print("TensorFlow is NOT using GPU.")
