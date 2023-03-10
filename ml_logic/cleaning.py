from ml_logic.params import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.data import Dataset
import os
import cv2
from google.cloud import storage
from google.oauth2 import service_account
import numpy as np
import matplotlib.pyplot as plt

def train_val_test_generator(source = SOURCE):
    """
    Generate the train, validation, and test batches.
    Converts the DirectoryIterator (dataset) from the ImageDataGenerator into
    X_images, y_target numpy arrays
    """
    def load_images(path, class_mode="categorical"):
        """
        Enter a path to load images from.
        """
        datagen = ImageDataGenerator(rescale = float(IMAGE_RESCALE_RATIO))
        images = datagen.flow_from_directory(path,
                                             target_size = (int(IMAGE_TARGET_WIDTH), int(IMAGE_TARGET_HEIGHT)),
                                             color_mode = "rgb",
                                             batch_size = int(BATCH_SIZE),
                                             class_mode = class_mode)

        return images

    def convert_DI_to_numpy(DI_dataset):
        """
        Converts DirectoryIterator dataset to numpy.array.
        Before cleaning images
        """

        # DI_dataset.reset()
        X_images = np.concatenate([DI_dataset.next()[0] for i in range(DI_dataset.__len__())])
        y_target = np.concatenate([DI_dataset.next()[1] for i in range(DI_dataset.__len__())])

        return X_images, y_target

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

    X_train, y_train = convert_DI_to_numpy(load_images(train_directory, class_mode="categorical"))
    X_val, y_val = convert_DI_to_numpy(load_images(val_directory, class_mode="categorical"))
    X_test, y_test = convert_DI_to_numpy(load_images(test_directory, class_mode="categorical"))

    return X_train, y_train, X_val, y_val, X_test, y_test

def preprocess_images(dataset):
    """
    Clean the images in the images i.e., X_train, X_val, X_test
    """
    def clean_images(image):
        """
        Input an image to add a rectangle to cover the green or black box on
        the resized and normalized image (-1 box)
        """
        # Identified ROI in resized and normalized image
        y1 = 148
        y2 = 224
        x1 = 0
        x2 = 77

        image_clean = cv2.rectangle(image, (x1,y1), (x2,y2),(-1,-1,-1),-1)

        return image_clean

    cleaned_X = []

    for i in range(dataset.shape[0]):
        cleaned_X.append(clean_images(dataset[i,:,:,:]))

    return cleaned_X

def convert_numpy_to_TFDataset(X, y):
        """
        Converts numpy.array back to TF format but this time a TF dataset.
        Before training model
        """
        dataset = Dataset.from_tensor_slices((X,y)).batch(int(BATCH_SIZE))

        return dataset

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_generator(source = SOURCE)
    print("Data split generated")
    # X_train_clean = preprocess_images(X_train)
    # print("Training images cleaned")
    # X_val_clean = preprocess_images(X_val)
    # print("Val images cleaned")
    X_test_clean = preprocess_images(X_test)
    print("finished")

    test_cleaned = convert_numpy_to_TFDataset(X_test, y_test)
    print("converted array back to TF")
    element_spec_X, element_spec_y = test_cleaned.element_spec
    print(element_spec_X.shape)
    print(element_spec_y.shape)
