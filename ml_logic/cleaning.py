from ml_logic.params import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from google.cloud import storage
from google.oauth2 import service_account
import numpy as np

def train_val_test_generator(source = SOURCE):
    """
    Generates X_train, y_train, X_val, y_val, X_test, y_test.
    """
    def load_images(path):
        """
        Enter a path to load images from.
        """
        datagen = ImageDataGenerator(rescale = float(IMAGE_RESCALE_RATIO))
        images = datagen.flow_from_directory(path,
                                             target_size = (int(IMAGE_TARGET_WIDTH), int(IMAGE_TARGET_HEIGHT)),
                                             color_mode = "rgb",
                                             batch_size = int(BATCH_SIZE),
                                             class_mode = "categorical")

        return images

    def convert_to_numpy(dataset):
        """
        Converts DirectoryIterator dataset to numpy.array before cleaning images.
        """
        X = np.concatenate([dataset.next()[0] for i in range(dataset.__len__())])
        y = np.concatenate([dataset.next()[1] for i in range(dataset.__len__())])

        return X, y

    if source == "local":
        train_directory = os.path.join(RAW_DATA_PATH, "train")
        val_directory = os.path.join(RAW_DATA_PATH, "val")
        test_directory = os.path.join(RAW_DATA_PATH, "test")

    #this doesn't work right now
    if source == "cloud":
        credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)
        client = storage.Client(project = GCLOUD_PROJECT_ID, credentials = credentials)
        bucket = client.get_bucket(BUCKET_NAME)

        train_directory = f"gs://{BUCKET_NAME}/train"
        val_directory = f"gs://{BUCKET_NAME}/val"
        test_directory = f"gs://{BUCKET_NAME}/test"

    X_train, y_train = convert_to_numpy(load_images(train_directory))
    X_val, y_val = convert_to_numpy(load_images(val_directory))
    X_test, y_test = convert_to_numpy(load_images(test_directory))

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_generator(source = SOURCE)
