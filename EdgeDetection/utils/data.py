import os
import numpy as np
import cv2
from scipy.ndimage import binary_dilation
from skimage.io import imread
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from typing import List, Tuple
from utils.features import extract_features_at_pixel

def get_image_gt_pairs(image_dir: str, gt_dir: str) -> List[Tuple[str, str]]:
    """Returns a list of (image, ground truth) file path pairs from the given directories."""
    image_files = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".bmp")
    ])
    gt_files = sorted([
        os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith(".bmp")
    ])

    assert len(image_files) == len(gt_files), "Mismatch in image and ground truth file counts"
    return list(zip(image_files, gt_files))

def load_image_rgb(path: str) -> np.ndarray:
    """Loads image in RGB format."""
    return cv2.imread(path)[:, :, ::-1]

def load_image_bw(path: str) -> np.ndarray:
    """Loads image in BW format."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def compute_ST_map(gray, sigma=1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes structor tensor  image in BW format."""
    Jxx, Jxy, Jyy = structure_tensor(gray, sigma=sigma)
    ST = np.array([Jxx, Jxy, Jyy])
    l1, l2 = structure_tensor_eigenvalues(ST)

    strength = np.sqrt(l1)
    coherence = (l1 - l2) / (l1 + l2 + 1e-12)
    orientation = 0.5 * np.arctan2(2*Jxy, Jxx - Jyy)
    
    return strength, coherence, orientation

def extract_labeled_dataset_from_image(
    image_rgb: np.ndarray,
    gt_image_rgb: np.ndarray,
    r: int = 3,
    ridge_margin: int = 3,
    ST_map: np.ndarray= None,
    image_bw: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts features (r x r patch and/or ST values) at pixels on Canny edges and labels them as ridge or not.

    Parameters:
    - image_rgb: (H, W, 3) RGB image
    - gt_image_rgb: (H, W, 3) Ground truth image with red ridge lines
    - r: radius of patch
    - ridge_margin: dilation iterations around red pixels to include as ridge

    Returns:
    - features: (N, number of features) array
    - labels: (N,) binary array
    """
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(image_gray, 50, 200)

    red_mask = (gt_image_rgb[:, :, 0] == 255) & (gt_image_rgb[:, :, 1] == 0) & (gt_image_rgb[:, :, 2] == 0)
    dilated_red_mask = binary_dilation(red_mask, iterations=ridge_margin)

    features_list, labels = [], []
    h, w = image_rgb.shape[:2]

    # Use grayscale image if specified
    if image_bw is not None:
        image = image_bw
    else:
        image = image_rgb

    for y in range(r, h - r):
        for x in range(r, w - r):
            if edges[y, x] == 0:
                continue

            features = extract_features_at_pixel(image, x, y, r=r, ST_map=ST_map) # Only use ST_map if specified
            label = int(dilated_red_mask[y, x])

            features_list.append(features)
            labels.append(label)
    
    return features_list, labels

def prepare_dataset(
    image_dir: str,
    gt_dir: str,
    r: int = 3,
    use_bw: bool = False,
    use_ST: bool = False,
    ridge_margin: int = 3,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads image/GT pairs, extracts labeled patches, and computes features.

    Parameters:
    - use_bw: If True, uses grayscale images for feature extraction.
    - use_ST: If True, includes structure tensor features.
    """
    pairs = get_image_gt_pairs(image_dir, gt_dir)
    X_list, y_list = [], []

    for i, (img_path, gt_path) in enumerate(pairs):
        image_rgb = load_image_rgb(img_path)
        gt_rgb = load_image_rgb(gt_path)

        # Load grayscale if requested
        image_gray = load_image_bw(img_path)
        ST_map = compute_ST_map(image_gray) if use_ST else None

        # Feature extraction image depends on use_bw
        image_input = image_gray if use_bw else image_rgb

        # Extract labeled features from edges
        features_list, labels = extract_labeled_dataset_from_image(
            image_rgb=image_rgb,
            gt_image_rgb=gt_rgb,
            r=r,
            ridge_margin=ridge_margin,
            ST_map=ST_map,
            image_bw=image_input
        )

        X_list.append(features_list)
        y_list.append(labels)

        if verbose:
            print(f"Processed {img_path} ({len(features_list)} data points)")

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    return X, y

def balance_dataset(X: np.ndarray, y: np.ndarray, random_seed: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Balances a binary-labeled dataset by undersampling the majority class.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    n_samples = min(len(pos_indices), len(neg_indices))

    selected_pos = np.random.choice(pos_indices, n_samples, replace=False)
    selected_neg = np.random.choice(neg_indices, n_samples, replace=False)

    selected_indices = np.concatenate([selected_pos, selected_neg])
    np.random.shuffle(selected_indices)

    return X[selected_indices], y[selected_indices]
