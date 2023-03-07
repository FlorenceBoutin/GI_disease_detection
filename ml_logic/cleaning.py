from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from google.cloud import storage

def train_val_test_generator():
    """
    Generate the train, validation, and test batches.
    """
    def load_images(folder):
        """
        Enter a folder directory with images to load.
        """
        datagen = ImageDataGenerator(rescale = 1. / 255)
        images = datagen.flow_from_directory(folder,
                                             target_size = (224, 224),
                                             color_mode = "rgb",
                                             batch_size = 32,
                                             class_mode = "categorical")
        return images

    bucket_name = os.environ.get("BUCKET_NAME")

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    train_directory = bucket.blob("train/*")
    val_directory = bucket.blob("val/*")
    test_directory = bucket.blob("test/*")

    # train_directory = f"{bucket}/train"
    # val_directory = f"{bucket}/val"
    # test_directory = f"{bucket}/test"

    train_dataset = load_images(train_directory)
    val_dataset = load_images(val_directory)
    test_dataset = load_images(test_directory)

    print(train_dataset, val_dataset, test_dataset)
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    print(train_val_test_generator())
