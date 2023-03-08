import os

# ---------- ENVIRONMENT VARIABLES ----------
SOURCE = os.environ.get("SOURCE")
RAW_DATA_PATH = os.environ.get("RAW_DATA_PATH")

GCLOUD_PROJECT_ID = os.environ.get("GCLOUD_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION")

BUCKET_NAME = os.environ.get("BUCKET_NAME")
GOOGLE_APPLICATION_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

# ---------------- CONSTANTS ----------------
IMAGE_RESCALE_RATIO = 1. / 255
IMAGE_TARGET_WIDTH = 224
IMAGE_TARGET_HEIGHT = 224
