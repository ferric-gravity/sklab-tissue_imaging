"""
Iteration 2: Improved ResNet Training and Evaluation
---------------------------------------------------
This script corresponds to Iteration-2 of the experiments for
Breast Cancer Classification using H&E stained images.

Key additions over Iteration-1:
- Modular training function
- Better ResNet fine-tuning
- Confusion matrix and classification report export
- Device-aware training (CPU / MPS / CUDA)
"""

from fastai.vision.all import *
from sklearn.metrics import classification_report
import pandas as pd
import torch

# --------------------
# Configuration
# --------------------
DATA_PATH = Path("data/ICIAR2018_BACH/Photos")
IMAGE_SIZE = 512
BATCH_SIZE = 16
VALID_PCT = 0.2
EPOCHS = 6
ARCH = "resnet26d"
MODEL_OUT = "resnet_iteration2.pkl"

# --------------------
# Device selection
# --------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# --------------------
# DataLoaders
# --------------------
dls = ImageDataLoaders.from_folder(
    DATA_PATH,
    valid_pct=VALID_PCT,
    seed=42,
    item_tfms=Resize(IMAGE_SIZE),
    batch_tfms=aug_transforms(),
    bs=BATCH_SIZE
)

# --------------------
# Training function
# --------------------
def train_model(arch: str, epochs: int):
    """
    Train a CNN architecture using fastai vision_learner.

    Parameters
    ----------
    arch : str
        Model architecture name (e.g., 'resnet26d')
    epochs : int
        Number of fine-tuning epochs
    """
    learn = vision_learner(dls, arch, metrics=error_rate)
    learn.to(DEVICE)
    learn.fine_tune(epochs, base_lr=1e-2)
    return learn

# --------------------
# Train model
# --------------------
learn = train_model(ARCH, EPOCHS)

# --------------------
# Save model
# --------------------
learn.export(MODEL_OUT)

# --------------------
# Evaluation
# --------------------
interp = ClassificationInterpretation.from_learner(learn)

# Confusion matrix
interp.plot_confusion_matrix(figsize=(6, 6))
plt.savefig("confusion_matrix.png", transparent=True, facecolor="white")

# Classification report
preds, targets = learn.get_preds()
report = classification_report(
    targets,
    preds.argmax(dim=1),
    target_names=dls.vocab,
    output_dict=True
)

df_report = pd.DataFrame(report).transpose()
df_report.to_csv("classification_report.csv")
