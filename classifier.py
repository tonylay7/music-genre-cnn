import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np


class Convolutional_NN():
    """A convolutional neural network

    Attributes:
        model: the model
        history: history data of the training 
    """
    def __init__(self,data_path):
        self.model = None
        self.history = None

    def build_model(self,input_shape):
        """ Builds a convolutional neural network

        Parameters:
            input_shape: the shape of the input to the neural network
        """
        self.model = keras.Sequential()

        # 1st convolutional layer
        self.model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        self.model.add(keras.layers.BatchNormalization())

        # 2nd convolutional layer
        self.model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        self.model.add(keras.layers.BatchNormalization())

        # 3rd convolutional layer
        self.model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
        self.model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
        self. model.add(keras.layers.BatchNormalization())

        # Flatten output and feed it into a dense layer
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(64, activation='relu'))
        self.model.add(keras.layers.Dropout(0.3))

        # Output layer
        self.model.add(keras.layers.Dense(10, activation='softmax'))

        # Compile model
        optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        self.model.compile(optimizer=optimiser,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        self.model.summary()

    def train_model(self, X_train, y_train, validation_data, batch_size=32, epochs=30):
        """Trains the convolutional neural network and provides history data

        Parameters:
            X_train: X data for training set
            y_train: y labels for training set
            validation_data: the validation data for training
            batch_size: the batch size for training
            epochs: number of epochs for training
        """
        self.history = self.model.fit(X_train, y_train, validation_data=validation_data, batch_size=batch_size, epochs=epochs)

    def predict(self, X, y):
        """Predict a single sample using the trained model

        Parameters:
            X: data for training sample
            y: label for training sample
        """

        # Add a dimension to input data for sample as model.predict() expects a 4d arra
        X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

        # Perform prediction
        prediction = self.model.predict(X)

        # Choose the index of the prediction with the largest probability
        predicted_index = np.argmax(prediction, axis=1)

        print("Target: {}, Predicted label: {}".format(y, predicted_index))

    def plot_history(self):
        """Plots accuracy/loss for training/validation set as a function of the epochs
        """
        _, axes = plt.subplots(2)

        # Create accuracy sublpot
        axes[0].plot(self.history.history["accuracy"], label="Train accuracy")
        axes[0].plot(self.history.history["val_accuracy"], label="Test accuracy")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend(loc="lower right")
        axes[0].set_title("Accuracy evaluation")

        # create error sublpot
        axes[1].plot(self.history.history["loss"], label="Train error")
        axes[1].plot(self.history.history["val_loss"], label="Test error")
        axes[1].set_ylabel("Error")
        axes[1].set_xlabel("Epoch")
        axes[1].legend(loc="upper right")
        axes[1].set_title("Error evaluation")

        plt.show()