"""
APTOS 0-4 DR classifier config: paths and hyperparameters.
"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Repo root (parent of classify); contains datasets/datasets
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Data root (parent of CROP and aptos2019). Override with env:
#   export APTOS_DATA_ROOT=/path/to/datasets/datasets
DATA_ROOT = os.environ.get(
    "APTOS_DATA_ROOT",
    os.path.join(REPO_ROOT, "datasets", "datasets"),
)

# CROP image dirs (aligned with baseline CROP_SPLIT_DIRS)
CROP_ROOT = os.path.join(DATA_ROOT, "CROP")
TRAIN_IMG_DIR = os.path.join(CROP_ROOT, "train_images", "train_images")
VAL_IMG_DIR = os.path.join(CROP_ROOT, "val_images", "val_images")
TEST_IMG_DIR = os.path.join(CROP_ROOT, "test_images", "test_images")

# Label CSVs
APTOS_ROOT = os.path.join(DATA_ROOT, "aptos2019")
TRAIN_CSV = os.path.join(APTOS_ROOT, "train_1.csv")
VAL_CSV = os.path.join(APTOS_ROOT, "valid.csv")
TEST_CSV = os.path.join(APTOS_ROOT, "test.csv")

# Training hyperparameters
NUM_CLASSES = 5
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224

# Output dirs
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
