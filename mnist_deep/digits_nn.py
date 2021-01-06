import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def show_image(img_data):
    """
    Show image for given matrix

    Args:
        img_data - 2d matrix with image data
    """

    plt.imshow(img_data, cmap = plt.cm.binary)
    plt.show()

def normalize_matrix(matrix):
    """
    For given matrix (or dataset of matrices), returns matrix
    (or dataset of matrices) with normalized values (float in range 0-1)

    Args:
        matrix - matrix (or dataset of matrices)
    Returns:
        normalized matrix
    """
    normalized = tf.keras.utils.normalize(matrix)
    return normalized


if __name__ == "__main__":

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # show_digit(x_train[0])
    x_train = normalize_matrix(x_train)
    x_test = normalize_matrix(x_test)

    # Model preparation
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer=tf.keras.optimizers.Adam.__name__, 
                loss=tf.keras.losses.SparseCategoricalCrossentropy.__name__,
                metrics=['accuracy'])

    # Model training
    model.fit(x_train, y_train, epochs=3)

    # Validation of the model
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss, val_acc) 

    ## Saving and loading model
    # model.save('nn_helloworld')
    # load_model=tf.keras.models.load_model('nn_helloworld')

    # Stores probability distributions for all elements in x_text
    predictions = model.predict([x_test])

    # Show sample prediction
    show_image(x_test[0])
    print(np.argmax(predictions[0]))
