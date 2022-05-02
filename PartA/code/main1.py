"""
This file is used for the network consisting of ONE hidden layer
"""

import os
from tensorflow.keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from preprocessing import get_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # avoid tesor flow warnings

NUM_OF_NODES = 0

def get_model(n_inputs, n_outputs):

    """
    Create and return a model based on the input and output size. Learning rate is by default 0.001.
    Initially it consists of an input layer, one hidden layer and an output layer.
    """

    model = Sequential()
    model.add(Dense(NUM_OF_NODES, input_dim=n_inputs, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["categorical_crossentropy", "mean_squared_error", "accuracy"])
    return model

def plot_result(history, fold, hidden_layers):

    """
    Plot the loss and accuracy of the model over the epochs.
    """

    metrics = [("loss", 0, 0), ("categorical_crossentropy", 0, 1), ("mean_squared_error", 1, 0), ("accuracy", 1, 1)]
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    for metric in metrics:
        axs[metric[1], metric[2]].plot(history.history[metric[0]], label=metric[0])
        axs[metric[1], metric[2]].plot(history.history["val_" + metric[0]], label="val_" + metric[0])
        axs[metric[1], metric[2]].set_title(metric[0], fontsize=10)
        axs[metric[1], metric[2]].set_ylabel(metric[0], fontsize=10)
        axs[metric[1], metric[2]].legend()
    axs[1, 0].set_xlabel("Epochs", fontsize=10)
    axs[1, 1].set_xlabel("Epochs", fontsize=10)
    fig.savefig("plots/fold_{}_hidden-layers_{}_hidden-nodes_{}.png".format(fold, hidden_layers, NUM_OF_NODES))

if __name__ == "__main__":
    
    # load data
    X, y = get_dataset("train-data.dat", "train-label.dat")
    X_test_diff, y_test_diff = get_dataset("test-data.dat", "test-label.dat")
    kf = KFold(n_splits=5)  # 5-fold cross validation
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    NUM_OF_NODES = n_inputs + n_outputs
    model = get_model(n_inputs, n_outputs)  # Train the model with 5-fold cross validation
    model.summary()
    epochs = 10

    for fold, (train_index, test_index) in enumerate(kf.split(X)):

        # Split data into train and test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print("Training model...")
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)

        # plot_result(history, "accuracy", fold)
        y_hat = model.predict(X_test_diff)
        y_hat = y_hat.round().astype("int")
        acc = accuracy_score(y_test_diff, y_hat)
        print('Fold {} with accuracy {}'.format((fold+1), acc))
        plot_result(history, fold, 1)