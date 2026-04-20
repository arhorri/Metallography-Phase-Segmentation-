"""
MetalDAM Dataset Preprocessing Pipeline for Semantic Segmentation
==================================================================
Author : Alireza Horri
Task   : Preprocess MetalDAM micrograph images + integer-label masks
         for training a U-Net (or similar) semantic segmentation model.

Environment : metallography (conda)
Dependencies: torch, torchvision, albumentations, opencv-python,
              Pillow, matplotlib, numpy, scikit-learn
"""

import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ──────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(os.environ.get("METALDAM_ROOT", Path(__file__).resolve().parent.parent))

CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────────
    "images_dir" : _PROJECT_ROOT / "data" / "MetalDAM" / "labeled" / "images",
    "labels_dir" : _PROJECT_ROOT / "data" / "MetalDAM" / "labeled" / "labels",
    "output_dir" : _PROJECT_ROOT / "data" / "MetalDAM" / "ready_for_training",

    # ── Preprocessing ──────────────────────────────────────────────────────────
    # Normalization mode: "minmax"  → scale to [0, 1]
    #                     "zscore"  → subtract mean, divide by std (ImageNet-like)
    "norm_mode"  : "minmax",
    # Standard mean/std used when norm_mode == "zscore" (single-channel grayscale)
    "norm_mean"  : 0.485,
    "norm_std"   : 0.229,

    # ── Augmentation ──────────────────────────────────────────────────────────
    "crop_size"  : 256,   # Random crop height == width (use 512 for higher-res)

    # ── Dataset Split ──────────────────────────────────────────────────────────
    "val_split"  : 0.20,  # 20 % validation, 80 % training
    "random_seed": 42,

    # ── DataLoader ─────────────────────────────────────────────────────────────
    "batch_size" : 4,
    "num_workers": 0,     # set > 0 on Linux; keep 0 on Windows to avoid spawn issues

    # ── Visualisation ──────────────────────────────────────────────────────────
    # One colour per class ID (extend the list if you have more than 4 classes)
    "class_colors": [
        "#000000",   # class 0 – background / martensite (black)
        "#e41a1c",   # class 1 – austenite (red)
        "#377eb8",   # class 2 – precipitates (blue)
        "#4daf4a",   # class 3 – inclusions (green)
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# 1.  FILE MATCHING UTILITY
# ──────────────────────────────────────────────────────────────────────────────

# Accepted image file extensions
IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def build_file_pairs(images_dir: str, labels_dir: str) -> list[tuple[Path, Path]]:
    """
    Scan *images_dir* and *labels_dir* and return a sorted list of
    (image_path, label_path) tuples where each pair shares the same stem.

    Raises
    ------
    FileNotFoundError
        If either directory does not exist.
    ValueError
        If any image file has no corresponding label (or vice-versa).
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    # ── Validate directories ───────────────────────────────────────────────────
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found:\n  {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found:\n  {labels_dir}")

    # ── Collect stems ─────────────────────────────────────────────────────────
    # Build stem → path maps (ignore non-image files)
    img_map  = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    lbl_map  = {p.stem: p for p in labels_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}

    img_stems = set(img_map.keys())
    lbl_stems = set(lbl_map.keys())

    # ── Cross-check ───────────────────────────────────────────────────────────
    missing_labels = img_stems - lbl_stems
    missing_images = lbl_stems - img_stems

    if missing_labels:
        raise ValueError(
            f"The following images have NO corresponding label:\n"
            + "\n".join(f"  {img_map[s]}" for s in sorted(missing_labels))
        )
    if missing_images:
        raise ValueError(
            f"The following labels have NO corresponding image:\n"
            + "\n".join(f"  {lbl_map[s]}" for s in sorted(missing_images))
        )

    # ── Build sorted pairs ────────────────────────────────────────────────────
    pairs = [(img_map[stem], lbl_map[stem]) for stem in sorted(img_stems)]
    print(f"[✓] Matched {len(pairs)} image–label pairs.")
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# 2.  FILE SPLIT → DISK
# ──────────────────────────────────────────────────────────────────────────────

def prepare_file_splits(
    pairs      : list[tuple[Path, Path]],
    output_dir : Path,
    val_split  : float = 0.20,
    random_seed: int   = 42,
    overwrite  : bool  = False,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """
    Split *pairs* into train / val and copy files into:

        output_dir/
          train/images/   train/labels/
          val/images/     val/labels/

    Returns
    -------
    train_pairs, val_pairs  — pointing to the new output locations.
    """
    train_raw, val_raw = train_test_split(
        pairs,
        test_size    = val_split,
        random_state = random_seed,
        shuffle      = True,
    )

    train_pairs, val_pairs = [], []

    for split, split_pairs, out_list in [
        ("train", train_raw, train_pairs),
        ("val",   val_raw,   val_pairs),
    ]:
        img_out = output_dir / split / "images"
        lbl_out = output_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_src, lbl_src in split_pairs:
            img_dst = img_out / img_src.name
            lbl_dst = lbl_out / lbl_src.name

            if overwrite or not img_dst.exists():
                shutil.copy2(img_src, img_dst)
            if overwrite or not lbl_dst.exists():
                shutil.copy2(lbl_src, lbl_dst)

            out_list.append((img_dst, lbl_dst))

        print(f"[✓] {split:5s} → {len(out_list):3d} pairs  ({img_out})")

    return train_pairs, val_pairs


# ──────────────────────────────────────────────────────────────────────────────
# 3.  IMAGE / MASK LOADING UTILITIES  (reads from ready_for_training/ paths)
# ──────────────────────────────────────────────────────────────────────────────

def load_image(path: Path, norm_mode: str, mean: float, std: float) -> np.ndarray:
    """
    Load a grayscale microstructure image and normalise it.

    Parameters
    ----------
    path      : Path to the image file.
    norm_mode : "minmax"  → output in [0.0, 1.0]  (float32)
                "zscore"  → (x - mean) / std       (float32)

    Returns
    -------
    np.ndarray of shape (H, W) with dtype float32.
    """
    # cv2.IMREAD_GRAYSCALE reads 8-bit; IMREAD_ANYDEPTH handles 16-bit TIFFs
    img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"OpenCV could not read image: {path}")

    img = img.astype(np.float32)

    if norm_mode == "minmax":
        lo, hi = img.min(), img.max()
        # Guard against blank/uniform images
        img = (img - lo) / (hi - lo + 1e-8)
    elif norm_mode == "zscore":
        img = (img - mean) / std
    else:
        raise ValueError(f"Unknown norm_mode '{norm_mode}'. Use 'minmax' or 'zscore'.")

    return img  # shape: (H, W)


def load_mask(path: Path) -> np.ndarray:
    """
    Load a single-channel label mask, preserving exact integer class IDs.

    The mask is intentionally loaded WITHOUT any interpolation or colour
    conversion.  PIL with mode "L" (8-bit) or "I" (32-bit int) keeps the raw
    integer values intact.

    Returns
    -------
    np.ndarray of shape (H, W) with dtype uint8 (class IDs 0-255).
    """
    # -- Preferred: PIL approach (zero risk of interpolation artefacts) --------
    mask_pil = Image.open(str(path))

    # Handle 16/32-bit TIFFs (mode "I" or "I;16") transparently
    if mask_pil.mode not in ("L", "P"):
        mask_pil = mask_pil.convert("L")

    mask = np.array(mask_pil, dtype=np.uint8)

    # -- Safety check: verify values are plausible class IDs ------------------
    unique_vals = np.unique(mask)
    if unique_vals.max() > 255:
        raise ValueError(
            f"Mask {path.name} contains pixel values > 255 ({unique_vals.max()})."
            " This likely indicates a multi-channel or float mask was loaded incorrectly."
        )

    return mask  # shape: (H, W), dtype uint8


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ALBUMENTATIONS AUGMENTATION PIPELINES
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transforms(crop_size: int) -> A.Compose:
    """
    Training augmentation pipeline tailored for metallographic microstructures.

    Key design decisions
    --------------------
    * RandomCrop / PadIfNeeded  – crops to a fixed spatial size so all batches
      are uniform.  Pad first if any image is smaller than crop_size.
    * Flips / Rotate90          – microstructures have no canonical orientation;
      all rotations are physically valid.
    * GridDistortion /
      ElasticTransform          – simulate specimen preparation artefacts and
      slight lens distortions without destroying grain boundaries.
    * NO colour jitter           – images are grayscale; brightness/contrast are
      kept mild to preserve phase-contrast differences that encode class info.

    The `additional_targets` dict tells Albumentations that "mask" is a mask so
    the *same* random parameters are applied to both image and mask.
    Crucially, masks use NEAREST interpolation internally to avoid corrupting
    integer class IDs.
    """
    return A.Compose(
        [
            # ── Spatial size normalisation ─────────────────────────────────────
            # Pad only if the image is smaller than the desired crop size
            A.PadIfNeeded(
                min_height=crop_size,
                min_width=crop_size,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            ),
            # Take a random crop of fixed spatial size
            A.RandomCrop(height=crop_size, width=crop_size, p=1.0),

            # ── Geometric augmentations ────────────────────────────────────────
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            # Elastic distortion – simulates slight specimen warping
            # interpolation=INTER_NEAREST for mask is handled internally by A
            A.ElasticTransform(
                alpha=120,
                sigma=6,
                p=0.3,
            ),
            # Grid distortion – global non-linear warp
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.15,
                p=0.3,
            ),

            # ── Intensity augmentations (grayscale-safe) ───────────────────────
            # Mild brightness / contrast variation to mimic illumination changes
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.4,
            ),
            # Gaussian blur – simulate slight focus variation
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),

            # ── Convert to PyTorch tensor ──────────────────────────────────────
            # Image  → float32 tensor of shape (1, H, W)  [channel-first]
            # Mask   → int64  tensor of shape (H, W)      [no channel dim]
            ToTensorV2(),
        ]
    )


def get_val_transforms(crop_size: int) -> A.Compose:
    """
    Validation pipeline: only deterministic spatial resizing (no random ops).
    We use CenterCrop to get the same spatial size as training without
    introducing randomness that would invalidate the evaluation.
    """
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=crop_size,
                min_width=crop_size,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            ),
            A.CenterCrop(height=crop_size, width=crop_size, p=1.0),
            ToTensorV2(),
        ]
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PYTORCH DATASET CLASS
# ──────────────────────────────────────────────────────────────────────────────

class MetalDAMDataset(Dataset):
    """
    PyTorch Dataset for the MetalDAM semantic segmentation benchmark.

    Parameters
    ----------
    pairs      : List of (image_path, mask_path) tuples (from build_file_pairs).
    transform  : Albumentations Compose pipeline (train or val).
    norm_mode  : Passed to load_image(); "minmax" or "zscore".
    norm_mean  : Used when norm_mode=="zscore".
    norm_std   : Used when norm_mode=="zscore".

    Returns (per __getitem__)
    -------------------------
    image : torch.Tensor  shape (1, H, W)  dtype float32
    mask  : torch.Tensor  shape (H, W)     dtype int64  ← class IDs
    """

    def __init__(
        self,
        pairs     : list[tuple[Path, Path]],
        transform : A.Compose,
        norm_mode : str   = "minmax",
        norm_mean : float = 0.485,
        norm_std  : float = 0.229,
    ):
        self.pairs     = pairs
        self.transform = transform
        self.norm_mode = norm_mode
        self.norm_mean = norm_mean
        self.norm_std  = norm_std

    # ── Required Dataset methods ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, lbl_path = self.pairs[idx]

        # 1. Load raw data
        image = load_image(img_path, self.norm_mode, self.norm_mean, self.norm_std)
        mask  = load_mask(lbl_path)

        # 2. Apply augmentations
        #    Albumentations expects:
        #      image → (H, W) or (H, W, C) numpy float32
        #      mask  → (H, W) numpy uint8 / int
        augmented = self.transform(image=image, mask=mask)
        image_t   = augmented["image"]   # shape: (1, H, W) float32 tensor
        mask_t    = augmented["mask"]    # shape: (H, W)    uint8  tensor

        # 3. Ensure correct dtypes for PyTorch loss functions
        #    image → float32  (already correct from ToTensorV2)
        #    mask  → int64    (CrossEntropyLoss expects Long / int64)
        image_t = image_t.float()
        mask_t  = mask_t.long()

        return image_t, mask_t

    # ── Convenience ───────────────────────────────────────────────────────────

    def get_class_ids(self) -> set[int]:
        """Scan all masks and return the set of unique class IDs present."""
        ids = set()
        for _, lbl_path in self.pairs:
            ids.update(np.unique(load_mask(lbl_path)).tolist())
        return ids


# ──────────────────────────────────────────────────────────────────────────────
# 6.  TRAIN / VAL SPLIT  (in-memory, used only if skipping file prep)
# ──────────────────────────────────────────────────────────────────────────────

def split_dataset(
    pairs      : list[tuple[Path, Path]],
    val_split  : float = 0.20,
    random_seed: int   = 42,
    crop_size  : int   = 256,
    norm_mode  : str   = "minmax",
    norm_mean  : float = 0.485,
    norm_std   : float = 0.229,
) -> tuple["MetalDAMDataset", "MetalDAMDataset"]:
    """
    Split *pairs* into train (80 %) and validation (20 %) MetalDAMDataset objects.

    Uses sklearn.train_test_split for a reproducible, stratification-free split
    (stratification is not practical at the image level for segmentation).

    Returns
    -------
    train_dataset, val_dataset  (both MetalDAMDataset instances)
    """
    train_pairs, val_pairs = train_test_split(
        pairs,
        test_size   = val_split,
        random_state= random_seed,
        shuffle     = True,
    )

    train_ds = MetalDAMDataset(
        pairs      = train_pairs,
        transform  = get_train_transforms(crop_size),
        norm_mode  = norm_mode,
        norm_mean  = norm_mean,
        norm_std   = norm_std,
    )
    val_ds = MetalDAMDataset(
        pairs      = val_pairs,
        transform  = get_val_transforms(crop_size),
        norm_mode  = norm_mode,
        norm_mean  = norm_mean,
        norm_std   = norm_std,
    )

    print(f"[✓] Train samples : {len(train_ds)}")
    print(f"[✓] Val   samples : {len(val_ds)}")
    return train_ds, val_ds


# ──────────────────────────────────────────────────────────────────────────────
# 7.  VERIFICATION / VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def build_colormap(hex_colors: list[str]) -> mcolors.ListedColormap:
    """
    Build a discrete matplotlib colormap from a list of hex colour strings.
    One colour per class ID.
    """
    rgb_colors = [mcolors.to_rgba(c) for c in hex_colors]
    return mcolors.ListedColormap(rgb_colors)


def visualise_batch(
    dataloader  : DataLoader,
    class_colors: list[str],
    num_samples : int = 4,
    save_path   : str | None = None,
) -> None:
    """
    Pull ONE batch from *dataloader*, apply a discrete colormap to each mask,
    and plot (augmented image | coloured mask) side-by-side for *num_samples*
    items.

    Parameters
    ----------
    dataloader   : A PyTorch DataLoader wrapping a MetalDAMDataset.
    class_colors : List of hex colour strings, one per class ID.
    num_samples  : How many samples from the batch to display (≤ batch_size).
    save_path    : If provided, save the figure to this file path.
    """
    # ── Fetch one batch ────────────────────────────────────────────────────────
    images, masks = next(iter(dataloader))
    # images : (B, 1, H, W) float32
    # masks  : (B, H, W)    int64

    n = min(num_samples, images.shape[0])
    cmap = build_colormap(class_colors)
    # The colormap boundary: 0 → n_classes
    n_classes  = len(class_colors)
    norm_cmap  = mcolors.BoundaryNorm(
        boundaries=np.arange(n_classes + 1) - 0.5,
        ncolors   =n_classes,
    )

    fig, axes = plt.subplots(
        nrows   = n,
        ncols   = 2,
        figsize = (8, 3.5 * n),
        dpi     = 120,
    )
    # Handle single-row edge case
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("MetalDAM – Augmented Images and Segmentation Masks", fontsize=13, y=1.01)

    for i in range(n):
        # ── Image ──────────────────────────────────────────────────────────────
        # Remove channel dimension: (1, H, W) → (H, W)
        img_np = images[i, 0].numpy()
        ax_img = axes[i, 0]
        ax_img.imshow(img_np, cmap="gray", vmin=img_np.min(), vmax=img_np.max())
        ax_img.set_title(f"Image #{i}  (aug.)", fontsize=9)
        ax_img.axis("off")

        # ── Mask ───────────────────────────────────────────────────────────────
        msk_np = masks[i].numpy()  # (H, W) int64
        unique_ids = np.unique(msk_np)
        ax_msk = axes[i, 1]
        im = ax_msk.imshow(msk_np, cmap=cmap, norm=norm_cmap, interpolation="nearest")
        ax_msk.set_title(
            f"Mask #{i}  (class IDs: {unique_ids.tolist()})", fontsize=9
        )
        ax_msk.axis("off")

        # Colourbar (shared per row)
        cbar = fig.colorbar(im, ax=ax_msk, fraction=0.046, pad=0.04)
        cbar.set_ticks(np.arange(n_classes))
        cbar.set_ticklabels([f"Class {c}" for c in range(n_classes)], fontsize=7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[✓] Verification figure saved → {save_path}")

    plt.show()
    print("[✓] Visualisation complete.")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  MetalDAM Preprocessing Pipeline")
    print("=" * 65)

    # ── Step 1: Match files ────────────────────────────────────────────────────
    pairs = build_file_pairs(CONFIG["images_dir"], CONFIG["labels_dir"])

    # ── Step 2: Copy files into ready_for_training/ directory structure ────────
    print("\n── Preparing output directories ───────────────────────────")
    train_pairs, val_pairs = prepare_file_splits(
        pairs       = pairs,
        output_dir  = CONFIG["output_dir"],
        val_split   = CONFIG["val_split"],
        random_seed = CONFIG["random_seed"],
    )

    # ── Step 3: Build PyTorch datasets from the output directories ─────────────
    train_ds = MetalDAMDataset(
        pairs     = train_pairs,
        transform = get_train_transforms(CONFIG["crop_size"]),
        norm_mode = CONFIG["norm_mode"],
        norm_mean = CONFIG["norm_mean"],
        norm_std  = CONFIG["norm_std"],
    )
    val_ds = MetalDAMDataset(
        pairs     = val_pairs,
        transform = get_val_transforms(CONFIG["crop_size"]),
        norm_mode = CONFIG["norm_mode"],
        norm_mean = CONFIG["norm_mean"],
        norm_std  = CONFIG["norm_std"],
    )

    # ── Step 4: Build DataLoaders ──────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size  = CONFIG["batch_size"],
        shuffle     = True,
        num_workers = CONFIG["num_workers"],
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = CONFIG["batch_size"],
        shuffle     = False,
        num_workers = CONFIG["num_workers"],
        pin_memory  = True,
    )

    print(f"\n[✓] Train DataLoader : {len(train_loader)} batches "
          f"(batch_size={CONFIG['batch_size']})")
    print(f"[✓] Val   DataLoader : {len(val_loader)} batches "
          f"(batch_size={CONFIG['batch_size']})")

    # ── Step 5: Quick sanity check on a single batch ───────────────────────────
    sample_imgs, sample_masks = next(iter(train_loader))
    print(f"\n── Batch shape check ──────────────────────────────────────")
    print(f"  images : {tuple(sample_imgs.shape)}   dtype={sample_imgs.dtype}")
    print(f"  masks  : {tuple(sample_masks.shape)}  dtype={sample_masks.dtype}")
    print(f"  image value range : [{sample_imgs.min():.3f}, {sample_imgs.max():.3f}]")
    print(f"  mask unique IDs   : {sample_masks.unique().tolist()}")

    # ── Step 6: Visualise one augmented batch ──────────────────────────────────
    print("\n── Visualisation ──────────────────────────────────────────")
    visualise_batch(
        dataloader   = train_loader,
        class_colors = CONFIG["class_colors"],
        num_samples  = min(4, CONFIG["batch_size"]),
        save_path    = str(CONFIG["output_dir"] / "metaldam_verification.png"),
    )

    print("\n[✓] Pipeline ready.  Pass `train_loader` and `val_loader` to your model.")
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()