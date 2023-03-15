from ml_logic.params import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.data import Dataset
import os
import cv2
from imblearn.over_sampling import SMOTE
from google.cloud import storage
from google.oauth2 import service_account
import numpy as np

def load_images(path, class_mode = "categorical", dataset_type = "train"):
    """
    Enter a path to load images from.

    class_mode should be "categorical" when we are training and validating our model.
    class_mode should be None when we are evaluating and predicting on new data.

    dataset_type will be either "train," "val," or "test."
    """
    if dataset_type == "train":
        datagen = ImageDataGenerator(rescale = float(IMAGE_RESCALE_RATIO),
                                     rotation_range = int(ROTATION_RANGE),
                                     shear_range = float(SHEAR_RANGE),
                                     zoom_range = float(ZOOM_RANGE),
                                     horizontal_flip = True,
                                     width_shift_range = float(WIDTH_SHIFT_RANGE),
                                     height_shift_range = float(HEIGHT_SHIFT_RANGE))
    if dataset_type == "val" or dataset_type == "test":
        datagen = ImageDataGenerator(rescale = float(IMAGE_RESCALE_RATIO))

    images = datagen.flow_from_directory(path,
                                         target_size = (int(IMAGE_TARGET_WIDTH), int(IMAGE_TARGET_HEIGHT)),
                                         color_mode = "rgb",
                                         batch_size = int(BATCH_SIZE),
                                         class_mode = class_mode)

    return images

def convert_DI_to_numpy(dataset):
    """
    Converts DirectoryIterator dataset to numpy.array.
    Returns X and y values for each dataset.
    """
    X = np.concatenate([dataset.next()[0] for i in range(dataset.__len__())])
    y = np.concatenate([dataset.next()[1] for i in range(dataset.__len__())])

    return X, y

def train_val_test_generator(source = SOURCE, class_mode = "categorical"):
    """
    class_mode should be "categorical" if run on existing data.
    Generates the train, validation, and test datasets, and the corresponding X and y for train, validation, and test.

    class_mode should be None if run on new data.
    Generates the numpy array for the uploaded image.
    """
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

    if class_mode == "categorical":
        X_train, y_train = convert_DI_to_numpy(load_images(train_directory, class_mode = class_mode, dataset_type = "train"))
        X_val, y_val = convert_DI_to_numpy(load_images(val_directory, class_mode = class_mode, dataset_type = "val"))
        X_test, y_test = convert_DI_to_numpy(load_images(test_directory, class_mode = class_mode, dataset_type = "test"))

        return X_train, y_train, X_val, y_val, X_test, y_test

    #need to figure out how to submit the image
    #update "new_image_directory" accordingly
    if class_mode == None:
        numpy_image = convert_numpy_to_TFDataset(load_images("new_image_directory", class_mode = class_mode))

        return numpy_image

def preprocess_images(X: np.array):
    """
    Clean the images in X_train, X_val, and X_test.
    """
    def clean_image(image: np.array):
        """
        Input an image to add a rectangle to cover the green or black box on the resized and normalized image (-1 box).
        """
        # Identified ROI for specific corner box in resized and normalized image
        image_clean = cv2.rectangle(image, (int(BOX_X1), int(BOX_Y1)), (int(BOX_X2), int(BOX_Y2)), (-1, -1, -1), -1)

        return image_clean

    cleaned_X = []

    for i in range(X.shape[0]):
        temp = (clean_image(X[i, :, :, :]))
        cleaned_X.append(cv2.resize(temp, (50,50))) # resized further to 50 x 50 image

    return cleaned_X

def convert_numpy_to_TFDataset(X, y):
    """
    Converts numpy.array back to TF format but this time a TF dataset.
    """
    dataset = Dataset.from_tensor_slices((X, y)).batch(int(BATCH_SIZE))

    return dataset

def pipeline(class_mode = "categorical"):
    """
    A pipeline of the entire cleaning process.
    If existing data, class_mode = "categorical".
    If new data, class_mode = None.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_generator(source = SOURCE, class_mode = class_mode)

    preprocessed_train = preprocess_images(X_train)
    preprocessed_val = preprocess_images(X_val)
    preprocessed_test = preprocess_images(X_test)

    train_dataset = convert_numpy_to_TFDataset(preprocessed_train, y_train)
    val_dataset = convert_numpy_to_TFDataset(preprocessed_val, y_val)
    test_dataset = convert_numpy_to_TFDataset(preprocessed_test, y_test)

    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    pipeline(class_mode = "categorical")

    # X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_generator(source = SOURCE)
    # print("Data split generated")
    # # X_train_clean = preprocess_images(X_train)
    # # print("Training images cleaned")
    # # X_val_clean = preprocess_images(X_val)
    # # print("Val images cleaned")
    # X_test_clean = preprocess_images(X_test)
    # print("finished")

    # test_cleaned = convert_numpy_to_TFDataset(X_test, y_test)
    # print("converted array back to TF")
    # element_spec_X, element_spec_y = test_cleaned.element_spec
    # print(element_spec_X.shape)
    # print(element_spec_y.shape)
