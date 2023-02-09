from classifier import Convolutional_NN
from dataset import load_yaml
from sklearn.model_selection import train_test_split
import json
import numpy as np

config = load_yaml()
DATA_PATH = config["data_path"]

def load_data(data_path):
    """Loads training dataset from json file.

    Parameters:
        data_path: Path to the json file containing data

    Returns:
        X (ndarray): inputs
        y (ndarray): targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y



def prepare_train_test_split(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.

    Parameters:
        test_size (float)- : Value in [0, 1] indicating percentage of data set to allocate to test split
        validation_size (float) - : Value in [0, 1] indicating percentage of train set to allocate to validation split

    Returns:
        X_train (ndarray): Input training set
        X_validation (ndarray): Input validation set
        X_test (ndarray): Input test set
        y_train (ndarray): Target training set
        y_validation (ndarray): Target validation set
        y_test (ndarray): Target test set
    """

    # Load data
    X, y = load_data(DATA_PATH)

    # Create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # Add an axis to input sets
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

if __name__ == "__main__":

    # Prepare train, validation and test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_train_test_split(0.25, 0.2)

    # Build and compile the CNN
    conv_net = Convolutional_NN(DATA_PATH)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)

    conv_net.build_model(input_shape)

    # Train the CNN
    conv_net.train_model(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)

    # Plot the accuracy/error for training and validation
    conv_net.plot_history()

    # Evaluate the model on the test set
    test_loss, test_acc = conv_net.model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # Predict a handful of samples
    for i in range(5):
        X_to_predict = X_test[i]
        y_to_predict = y_test[i]
        conv_net.predict(X_to_predict, y_to_predict)

