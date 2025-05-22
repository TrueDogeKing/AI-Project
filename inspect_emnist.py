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
MAT_PATH = "dataset/emnist-byclass.mat"
mat = loadmat(MAT_PATH)

# The EMNIST .mat template always stores the data under mat['dataset']
dataset = mat["dataset"]

# Helper to extract arrays
def get(split, field):
    return dataset[split][0, 0][field][0, 0]

# Helper to pull arrays out of the MATLAB struct

x_train = get("train", "images")  # shape: (124800, 784)
y_train = get("train", "labels")  # shape: (124800, 1)

# ---- 2. Reshape & normalize -------------------------------------------------
x_train = get("train", "images").reshape(-1, 28, 28).astype(np.float32) / 255.0
y_train = get("train", "labels").flatten()

mapping = dataset["mapping"][0, 0]  # shape (62, 2): [label_id, unicode]
# EMNIST Letters labels: 1 → ‘a’, …, 26 → ‘z’
label_to_char = {label: chr(codepoint) for label, codepoint in mapping}

# Optional reverse mapping: character → label index
char_to_label = {v: k for k, v in label_to_char.items()}

# ---- 3. Visualize a small grid ---------------------------------------------
fig, axes = plt.subplots(8, 8, figsize=(6, 6))
for ax, img, label in zip(axes.ravel(), x_train[:64], y_train[:64]):
    ax.imshow(img.T, cmap="gray")
    ax.set_title(label_to_char[label])
    ax.axis("off")

plt.tight_layout()
plt.show()
