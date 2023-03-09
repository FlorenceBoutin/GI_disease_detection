from ml_logic.params import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from google.cloud import storage
from google.oauth2 import service_account
import tqdm
import numpy as np

def train_val_test_generator(source = SOURCE):
    """
    Generate the train, validation, and test batches.
    """
    def load_images(path):
        """
        Enter a path to load images from.
        """
        datagen = ImageDataGenerator(rescale = IMAGE_RESCALE_RATIO)
        images = datagen.flow_from_directory(path,
                                             target_size = (IMAGE_TARGET_WIDTH, IMAGE_TARGET_HEIGHT),
                                             color_mode = "rgb",
                                             batch_size = BATCH_SIZE,
                                             class_mode = "categorical")

        return images

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

    train_dataset = load_images(train_directory)
    val_dataset = load_images(val_directory)
    test_dataset = load_images(test_directory)

    train_dataset.reset()
    X_train, y_train = next(train_dataset)
    for i in tqdm.tqdm(range(int(train_dataset.n / BATCH_SIZE) - 1)):
      img, label = next(train_dataset)
      X_train = np.append(X_train, img, axis = 0)
      y_train = np.append(y_train, label, axis = 0)

    val_dataset.reset()
    X_train, y_train = next(val_dataset)
    for i in tqdm.tqdm(range(int(val_dataset.n / BATCH_SIZE) - 1)):
      img, label = next(val_dataset)
      X_val = np.append(X_train, img, axis = 0)
      y_val = np.append(y_train, label, axis = 0)

    test_dataset.reset()
    X_train, y_train = next(test_dataset)
    for i in tqdm.tqdm(range(int(test_dataset.n / BATCH_SIZE) - 1)):
      img, label = next(test_dataset)
      X_test = np.append(X_train, img, axis = 0)
      y_test = np.append(y_train, label, axis = 0)

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_generator(source = SOURCE)
