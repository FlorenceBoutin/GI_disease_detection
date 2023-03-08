import os
from google.cloud import storage
from ml_logic.cleaning import load_images, train_val_test_generator

def get_data_from_cloud():
    """
    Retrieve data from Google Big Query.
    """
    bucket_name = os.environ.get("BUCKET_NAME")

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    images = load_images(bucket)
    train_dataset, val_dataset, test_dataset = train_val_test_generator(images)

    return train_dataset, val_dataset, test_dataset
