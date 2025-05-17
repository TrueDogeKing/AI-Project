"""
Quick viewer for the EMNIST Letters dataset (.mat file).

Requirements:
pip install scipy matplotlib numpy
"""

import string
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ---- 1. Load the .mat file --------------------------------------------------
MAT_PATH = "dataset/emnist-letters.mat"
mat = loadmat(MAT_PATH)

# The EMNIST .mat template always stores the data under mat['dataset']
dataset = mat["dataset"]

# Helper to pull arrays out of the MATLAB struct
def get(split, field):
    # e.g. split='train', field='images'
    return dataset[split][0, 0][field][0, 0]

x_train = get("train", "images")  # shape: (124800, 784)
y_train = get("train", "labels")  # shape: (124800, 1)

# ---- 2. Reshape & normalize -------------------------------------------------
x_train = x_train.reshape(-1, 28, 28).astype(np.float32) / 255.0
y_train = y_train.flatten().astype(np.int64)  # labels are 1–26

# EMNIST Letters labels: 1 → ‘a’, …, 26 → ‘z’
idx_to_char = dict(zip(range(1, 27), string.ascii_lowercase))

# ---- 3. Visualize a small grid ---------------------------------------------
samples = 64
fig, axes = plt.subplots(8, 8, figsize=(6, 6))
for ax, img, label in zip(axes.ravel(), x_train[:samples], y_train[:samples]):
    ax.imshow(img.T, cmap="gray")        # transpose fixes 90° rotation
    ax.set_title(idx_to_char[label])
    ax.axis("off")

plt.tight_layout()
plt.show()
