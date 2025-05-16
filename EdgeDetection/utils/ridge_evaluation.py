import numpy as np
import time
from scipy.spatial.distance import cosine
import cv2
from tqdm import tqdm

def average_pixel_distance(pred_line: np.ndarray, gt_line: np.ndarray) -> float:
    """
    Computes average vertical pixel distance between predicted and GT ridge lines.
    """
    assert len(pred_line) == len(gt_line), "Lines must be of equal length"
    return np.mean(np.abs(pred_line - gt_line))


def cosine_similarity_line(pred_line: np.ndarray, gt_line: np.ndarray) -> float:
    """
    Computes cosine similarity between the predicted and ground truth lines.
    """
    assert len(pred_line) == len(gt_line), "Lines must be of equal length"
    return 1 - cosine(pred_line, gt_line)  # Cosine similarity = 1 - cosine distance


def evaluate_ridge_predictions(image_paths: list, gt_paths: list, model, find_ridge_fn, r=3, use_ST=True, use_bw=True) -> dict:
    """
    Evaluates ridge predictions over a dataset.
    Returns average pixel distance, cosine similarity, and avg inference time.

    Parameters:
        - image_paths: list of paths to test images
        - gt_paths: list of paths to GT images (with red ridge line)
        - model: trained PyTorch model
        - find_ridge_fn: function to compute predicted ridge (e.g., find_ridge_DP_Canny or find_ridge_greedy)
        - r: patch radius
        - use_ST: whether to compute structure tensor

    Returns:
        - Dictionary with average pixel distance, cosine similarity, inference time
    """
    from utils.data import compute_ST_map
    results = {
        "avg_pixel_distance": [],
        "cosine_similarity": [],
        "inference_times": []
    }

    for img_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths), desc="Evaluating"):
        image = cv2.imread(img_path)
        image_input = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if use_bw else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if use_ST:
            ST_map = compute_ST_map(image_gray)
        else:
            ST_map = None

        # Ground truth ridge: extract from red pixels
        gt_img = cv2.imread(gt_path)
        red_mask = (gt_img[:, :, 2] == 255) & (gt_img[:, :, 1] == 0) & (gt_img[:, :, 0] == 0)
        gt_ridge = np.argmax(red_mask, axis=0)

        # Predict ridge and measure time
        start = time.time()
        pred_ridge = find_ridge_fn(image_input, model, ST_map=ST_map, r=r) 
        duration = time.time() - start

        if len(pred_ridge) != len(gt_ridge):
            min_len = min(len(pred_ridge), len(gt_ridge))
            pred_ridge = pred_ridge[:min_len]
            gt_ridge = gt_ridge[:min_len]

        # Calculate metrics
        results["avg_pixel_distance"].append(average_pixel_distance(pred_ridge, gt_ridge))
        results["cosine_similarity"].append(cosine_similarity_line(pred_ridge, gt_ridge))
        results["inference_times"].append(duration)

    # Combine results
    return {
        "avg_pixel_distance": np.mean(results["avg_pixel_distance"]),
        "cosine_similarity": np.mean(results["cosine_similarity"]),
        "avg_inference_time": np.mean(results["inference_times"])
    }