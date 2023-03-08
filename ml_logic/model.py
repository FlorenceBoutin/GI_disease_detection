import numpy as np
from tensorflow import keras

from keras import optimizers, Model, Sequential, layers
from keras.callbacks import EarlyStopping

def dummy_model():
    """
    Baseline prediction that always predicts the same value.
    Used to create the API while waiting for the first model to be ready.
    """
    return 1

def initialize_baseline_model(input_shape=(224, 224, 3)) -> Model:
    """
    Initialize the Neural Network with random weights
    """

    # Create instance of to model
    model = Sequential()

    # Add Convolution layers + Pooling and Dropout layers to limit overfitting.
    model.add(layers.Conv2D(64, kernel_size=(3,3), input_shape=input_shape, activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(3,3)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(32, kernel_size=(2,2), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(16, kernel_size=(2,2), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))

    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(15, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(5, activation='relu'))

    # Output layer
    model.add(layers.Dense(3, activation='softmax'))

    print("✅ baseline model initialized")

    return model


def compile_baseline_model(model: Model, learning_rate=0.001) -> Model:
    """
    Compile the Neural Network
    """
    recall = keras.metrics.Recall()
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
               optimizer=optimizer,
               metrics=[recall, 'accuracy'])

    print("✅ model compiled")
    return model

def train_baseline_model(model: Model,
                         train_data,
                         validation_data,
                         patience=5) :
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=0
    )

    history = model.fit(train_data,
        validation_data=validation_data,
        epochs=30,
        callbacks=[es],
        verbose=1)

    print(f"✅ model trained with accuracy of {round(np.max(history.history['val_accuracy']), 2)} and a recall of {round(np.max(history.history['val_recall']), 2)}.")

    return model, history
