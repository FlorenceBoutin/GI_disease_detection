from ml_logic.params import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from google.cloud import storage
from google.oauth2 import service_account

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
                                             batch_size = 64,
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

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    train_val_test_generator()
