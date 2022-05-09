"""
This file is used for the network consisting of ONE hidden layer.
"""

import os
import pathlib
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from preprocessing import get_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # avoid tesor flow warnings

NUM_OF_HIDDEN_LAYERS = 1
NUM_OF_NODES_HIDDEN_1 = 0

def get_model(n_inputs, n_outputs):

    """
    Create and return a model based on the input and output size. Learning rate is by default 0.001.
    Initially it consists of an input layer, one hidden layer and an output layer.
    """

    model = Sequential()
    model.add(Dense(NUM_OF_NODES_HIDDEN_1, input_dim=n_inputs, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))

    optimizer = optimizers.SGD(learning_rate=0.001)
    metrics = ["binary_crossentropy", "mean_squared_error", "accuracy"]
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    return model

def plot_result(metric, list1, list2, hidden_layers):

    """
    Plot the metrics of the model over the epochs.
    """
    fig, ax = plt.subplots()
    ax.plot(list1, label=metric)
    ax.plot(list2, label="val_" + metric)
    ax.legend()
    ax.set_title(metric + " for {} hidden layer nodes and 5 epochs".format(NUM_OF_NODES_HIDDEN_1), fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_xlabel("Epochs", fontsize=10)
    fig.savefig("../plots/{}_hidden-layers_{}_hidden-nodes_{}.png".format(metric, hidden_layers, NUM_OF_NODES_HIDDEN_1))

if __name__ == "__main__":
    
    # Load data
    X, y = get_dataset("train-data.dat", "train-label.dat")
    X_test, y_test = get_dataset("test-data.dat", "test-label.dat")

    kf = KFold(n_splits=5)
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    hidden_layer_nodes = [n_outputs, (n_outputs + n_inputs) // 2, n_inputs + n_outputs]
    NUM_OF_NODES_HIDDEN_1 = hidden_layer_nodes[0]
    model = get_model(n_inputs, n_outputs)
    model.summary()
    epochs = 10
    binary_cross_entropy_all = [0] * epochs
    binary_cross_entropy_val_all = [0] * epochs
    mean_squared_error_all = [0] * epochs
    mean_squared_error_val_all = [0] * epochs
    accuracy_all = [0] * epochs
    accuracy_val_all = [0] * epochs

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)

        binary_cross_entropy_all = [a + b for a, b in zip(binary_cross_entropy_all, history.history["binary_crossentropy"])]
        binary_cross_entropy_val_all = [a + b for a, b in zip(binary_cross_entropy_val_all, history.history["val_binary_crossentropy"])]
        mean_squared_error_all = [a + b for a, b in zip(mean_squared_error_all, history.history["mean_squared_error"])]
        mean_squared_error_val_all = [a + b for a, b in zip(mean_squared_error_val_all, history.history["val_mean_squared_error"])]
        accuracy_all = [a + b for a, b in zip(accuracy_all, history.history["accuracy"])]
        accuracy_val_all = [a + b for a, b in zip(accuracy_val_all, history.history["val_accuracy"])]

    binary_cross_entropy_all = [a / 5 for a in binary_cross_entropy_all]
    binary_cross_entropy_val_all = [a / 5 for a in binary_cross_entropy_val_all]
    mean_squared_error_all = [a / 5 for a in mean_squared_error_all]
    mean_squared_error_val_all = [a / 5 for a in mean_squared_error_val_all]
    accuracy_all = [a / 5 for a in accuracy_all]
    accuracy_val_all = [a / 5 for a in accuracy_val_all]

    plot_result("binary_crossentropy", binary_cross_entropy_all, binary_cross_entropy_val_all, NUM_OF_HIDDEN_LAYERS)
    plot_result("mean_squared_error", mean_squared_error_all, mean_squared_error_val_all, NUM_OF_HIDDEN_LAYERS)
    plot_result("accuracy", accuracy_all, accuracy_val_all, NUM_OF_HIDDEN_LAYERS)
    results = model.evaluate(X_test, y_test)  # Test the model using data never used for training
    print(results)