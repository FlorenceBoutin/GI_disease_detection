import glob
import os
import time
import pickle
from tensorflow import keras

def save_results(metrics: dict, registry_path) -> None:
    """
    Persist metrics locally on hard drive at
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(registry_path, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


def save_model(model: keras.Model, registry_path) -> None:
    """
    Persist trained model locally on hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # save model locally
    model_path = os.path.join(registry_path, "models", f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    # if MODEL_TARGET == "gcs":
    #     # 🎁 We give you this piece of code as a gift. Please read it carefully! Add breakpoint if you need!
    #     from google.cloud import storage

    #     model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
    #     client = storage.Client()
    #     bucket = client.bucket(BUCKET_NAME)
    #     blob = bucket.blob(f"models/{model_filename}")
    #     blob.upload_from_filename(model_path)

    #     print("✅ Model saved to gcs")
    #     return None

    return None
