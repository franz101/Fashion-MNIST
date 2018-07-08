def load_fashion_mnist(path, kind='train'):
    import os
    import numpy as np
    """Load MNIST data"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels

def normalize(X):
    import numpy as np
    X = X.reshape(X.shape[0], 28, 28, 1)
    X = X.astype('float32')
    X /= 255
    return X