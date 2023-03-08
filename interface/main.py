import numpy as np
from ml_logic.cleaning import train_val_test_generator
from ml_logic.model import *
from ml_logic.registry import *

def train(data_path,
          learning_rate=0.001,
          patience = 5) -> float:
    """
    - Load raw data from the path given
    - Train on the train dataset
    - Store training results and model weights

    Return val_accuracy and val_recall as float
    """

    train_dataset, val_dataset, test_dataset = train_val_test_generator(data_path)


    # Initialize baseline model
    model = initialize_baseline_model()

    # Compile the model
    model = compile_baseline_model(model, learning_rate=learning_rate)

    # Train the model
    model, history = train_baseline_model(model,
                                          train_data=train_dataset,
                                          validation_data=val_dataset,
                                          patience = patience)



    val_accuracy = np.min(history.history['val_accuracy'])
    val_recall = np.min(history.history['val_recall'])

    metrics = dict(
        accuracy = val_accuracy,
        recall = val_recall
    )

    # Save results on hard drive using taxifare.ml_logic.registry
    save_results(metrics=metrics)

    # Save model weight on hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")
    return metrics
