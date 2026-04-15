import os
import cv2
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

def match_image_label_pairs(image_dir: str, label_dir: str):
    """
    Scans the image and label directories and pairs them perfectly based on their base filenames.
    Raises ValueError if there is any mismatch (meaning an image is missing a label or vice versa).
    """
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        raise FileNotFoundError(f"One or both directories do not exist:\n{image_dir}\n{label_dir}")

    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))

    # Strip extensions to match just the base names 
    # (in case images are .jpg and labels are .png, though they usually match)
    image_names = {os.path.splitext(f)[0]: f for f in image_files if os.path.isfile(os.path.join(image_dir, f))}
    label_names = {os.path.splitext(f)[0]: f for f in label_files if os.path.isfile(os.path.join(label_dir, f))}

    if set(image_names.keys()) != set(label_names.keys()):
        missing_in_labels = set(image_names.keys()) - set(label_names.keys())
        missing_in_images = set(label_names.keys()) - set(image_names.keys())
        error_msg = ("Mismatch in file pairs!\n"
                     f"Images without labels: {list(missing_in_labels)[:5]}\n"
                     f"Labels without images: {list(missing_in_images)[:5]}")
        raise ValueError(error_msg)

    # Reconstruct exact paths for the pairs
    file_pairs = []
    for base_name in sorted(image_names.keys()):
        img_path = os.path.join(image_dir, image_names[base_name])
        lbl_path = os.path.join(label_dir, label_names[base_name])
        file_pairs.append((img_path, lbl_path))

    return file_pairs


def get_transforms(phase="train", crop_size=256):
    """
    Generates an albumentations augmentation pipeline tailored for metallurgical microstructures.
    The targets (image, mask) are processed identically by Albumentations.
    """
    if phase == "train":
        return A.Compose([
            A.RandomCrop(width=crop_size, height=crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.3),
            A.Normalize(mean=(0.5,), std=(0.5,)), # Standard 0-1 normalization for grayscale maps to -1 to +1
            ToTensorV2()
        ])
    else:
        # Validation just crops centrally and normalizes to match network inputs
        return A.Compose([
            A.CenterCrop(width=crop_size, height=crop_size),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])


class MetalDAMDataset(Dataset):
    """
    Custom PyTorch Dataset for loading grayscale images and integer-labeled masks.
    """
    def __init__(self, file_pairs, transform=None):
        """
        Args:
            file_pairs: List of tuples -> (image_path, label_path).
            transform: Albumentations Compose object.
        """
        self.file_pairs = file_pairs
        self.transform = transform

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.file_pairs[idx]

        # Load microstructure image as grayscale
        # Using cv2.IMREAD_GRAYSCALE returns (H, W) array of [0, 255]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Failed to load image at: {img_path}")
        
        # Albumentations standard grayscale single-channel expectation: (H, W, 1) or just (H, W)
        # We expand dims so to_tensor() keeps 3 spatial axes (1, H, W).
        image = np.expand_dims(image, axis=-1)

        # Load mask EXACTLY as it is, preserving exact integer values representing classes
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise IOError(f"Failed to load mask at: {mask_path}")

        # Apply composed augmentations
        if self.transform:
            # Albumentations smoothly preserves nearest-neighbor interpolation on 
            # integer masks behind the scenes when passed as 'mask'.
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Ensure mask is typed PyTorch Long (int64) for loss functions like CrossEntropyLoss
        mask = mask.to(torch.long)

        return image, mask


def get_dataloaders(image_dir, label_dir, batch_size=4, crop_size=256, val_split=0.2, random_state=42):
    """
    Splits the paired dataset into train and validation splits, returns them wrapped in PyTorch DataLoaders.
    """
    file_pairs = match_image_label_pairs(image_dir, label_dir)
    
    # Create the 80/20 train/val split
    train_pairs, val_pairs = train_test_split(file_pairs, test_size=val_split, random_state=random_state)
    
    train_dataset = MetalDAMDataset(train_pairs, transform=get_transforms("train", crop_size))
    val_dataset = MetalDAMDataset(val_pairs, transform=get_transforms("val", crop_size))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


def visualize_batch(dataloader, num_samples=4):
    """
    Fetches one batch and visually plots the Augmented Grayscale Images alongside their exact integer masks.
    We apply a discrete colormap to the integer mask for explicit visualization.
    """
    # Fetch a single batch from iterator
    images, masks = next(iter(dataloader))
    
    batch_size = images.size(0)
    samples_to_plot = min(batch_size, num_samples)
    
    fig, axes = plt.subplots(samples_to_plot, 2, figsize=(10, 5 * samples_to_plot))
    
    if samples_to_plot == 1:
        axes = np.expand_dims(axes, axis=0) # ensure 2D indexing safe
        
    # We use a distinct colormap for semantic labels
    cmap = plt.get_cmap('tab10')
        
    for i in range(samples_to_plot):
        img_tensor = images[i]
        mask_tensor = masks[i]
        
        # Denormalize image: Image (1, H, W) is normalized: mean=0.5, std=0.5 -> (x - 0.5)/0.5
        # So we reconstruct using: x = x * std + mean
        img_np = img_tensor.permute(1, 2, 0).numpy() # Shape: (H, W, 1)
        img_np = (img_np * 0.5) + 0.5
        img_np = np.clip(img_np, 0, 1) # back to pure 0-1 ranges
        
        # Squeeze mask for visualization: (H, W) integers
        mask_np = mask_tensor.numpy()
        
        # Image Plot
        ax_img = axes[i, 0]
        ax_img.imshow(img_np.squeeze(), cmap='gray')
        ax_img.set_title(f"Augmented Microstructure {i+1}")
        ax_img.axis('off')
        
        # Mask Plot
        ax_msk = axes[i, 1]
        # vmin/vmax fixed slightly higher than generic classes assuming typical semantic count < 10
        im = ax_msk.imshow(mask_np, cmap=cmap, vmin=0, vmax=9, interpolation='nearest') 
        ax_msk.set_title(f"Augmented Mask {i+1} (Classes)")
        ax_msk.axis('off')
        
        # Output color bar directly on the first mask to ensure values look correct
        if i == 0:
            cbar = fig.colorbar(im, ax=ax_msk, fraction=0.046, pad=0.04)
            cbar.set_label('Class ID')
            
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Determine absolute base directories mapping nicely across WSL/native environment 
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGE_DIR = os.path.join(BASE_DIR, "data", "MetalDAM", "labeled", "images")
    LABEL_DIR = os.path.join(BASE_DIR, "data", "MetalDAM", "labeled", "labels")

    print(f"Loading data from:\nImages: {IMAGE_DIR}\nLabels: {LABEL_DIR}")
    
    try:
        train_loader, val_loader = get_dataloaders(IMAGE_DIR, LABEL_DIR, batch_size=4)
        print(f"Data split successful:")
        print(f" -> Train Batches: {len(train_loader)} (approx {len(train_loader.dataset)} samples)")
        print(f" -> Val Batches: {len(val_loader)} (approx {len(val_loader.dataset)} samples)")
        
        print("\nVisualizing a batch from the training loader...")
        visualize_batch(train_loader, num_samples=3)
        
    except FileNotFoundError as err:
        print(f"\n[Warning]: {err}")
        print("Dataset missing directories. Pipeline is ready but waiting for data to populate `data/MetalDAM/labeled/`")
    except ValueError as err:
        print(f"\n[Error]: Dataset mismatch -> {err}")
    except StopIteration:
        print("\n[Warning]: Dataloader is empty, likely no images existed in directory.")
