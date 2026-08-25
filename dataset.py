"""
dataset.py

Loads the BloodMNIST dataset, converts it to normalized NumPy tensors, 
and caches the result to disk so downsteam scripts (train.py, 
evaluate.py) don't need to re-decode images on every run.

Source: BloodMNIST is distributed via the MedMNIST v2 benchmark collection
(Yang et al., 2023, Scientific Data — https://medmnist.com/), built from
peripheral blood cell microscopy images originally published by
Acevedo et al. (2020, Data in Brief), collected from healthy donors free
of infection, hematologic, or oncologic disease.

Note: per MedMNIST's own documentation, this dataset is not intended for
clinical use.
"""


import numpy as np 
import matplotlib.pyplot as plt
from medmnist import BloodMNIST


# Class order, verified against MedMNIST's own label mapping
CLASS_NAMES = [
    "basophil",
    "eosinophil",
    "erythroblast",
    "imature_granulocyte",
    "lymphocyte",
    "monocyte",
    "neutrophil",
    "platalet",
]


def load_bloodmnist(split: str):
    """
    Load one split of BloodMNIST and convert it to a PyTorch-convention
    NumPy array.

    Args:
        split: one of "train", "val", "test".

    
    Returns:
        images: np.ndarray, shape (N, 3, 28, 28), float32, values in [0,1]
        labels: np.ndarray, shape (N,), int64 class indices (0-7)
    """

    # download=True caches the .npz locally, so this only fetches from the network once, not on every run
    dataset = BloodMNIST(split=split, download=True)

    # Squezing it down to (N,) to match what CrossEntropyLoss expects later
    images = dataset.imgs
    labels = dataset.labels.squeeze()

    # Normalize pixel values from [0, 255] to [0, 1]
    # Reorder axes from channel-last to channel-first
    images = images.astype(np.float32) / 255.0
    images = images.transpose(0, 3, 1, 2)
    labels = labels.astype(np.int64)


    return images, labels



def save_class_distribution_plot(labels: np.ndarray, out_path:str):
    """
    Building a bar chart of class frequency in the training set and
    saving it to out_path. This documents the class imbalance that
    motivates the weighted CrossEntropyLoss used in train.py.

    Args:
        labels: np.ndarray of integer class indices
        out_path: file path to save the figure to, e.g. "assets/class_distribution.png"
    """

    counts = np.bincount(labels, minlength=len(CLASS_NAMES))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(CLASS_NAMES, counts, color="#4C72B0")
    ax.set_xlabel("Cell type")
    ax.set_ylabel("Number of images")
    ax.set_title("BloodMNIST class distribution (training set)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved class distribution plot to {out_path} "
          f"(min: {counts.min()}, max: {counts.max()}, "
          f"imbalance ration: {counts.max() / counts.min():.2f}x)")


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        images, labels = load_bloodmnist(split)
        print(f"[{split}] images: {images.shape}, dtype: {images.dtype} | "
              f"labels: {labels.shape}, dtype: {labels.dtype}")

        np.save(f"blood_{split}_images.npy", images)
        np.save(f"blood_{split}_labels.npy", labels)

    
    train_labels = np.load("blood_train_labels.npy")
    save_class_distribution_plot(labels, "assets/class_distribution.png")

