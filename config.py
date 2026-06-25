import torch
from pathlib import Path

# =========================
# PROJECT PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent

TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

CHECKPOINT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# TRAINING CONFIG
# =========================
IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 1e-4

NUM_CLASSES = 2

NUM_WORKERS = 0

PIN_MEMORY = True

# =========================
# DEVICE
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# MODEL FILES
# =========================
BEST_MODEL = MODEL_DIR / "best_model.pth"

LAST_CHECKPOINT = CHECKPOINT_DIR / "last_checkpoint.pth"

print("=" * 50)
print("DEVICE :", DEVICE)
print("TRAIN :", TRAIN_DIR)
print("TEST  :", TEST_DIR)
print("=" * 50)