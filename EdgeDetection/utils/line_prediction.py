import numpy as np
import cv2
import torch
from utils.features import extract_features_at_pixel

def generate_probability_map(image: np.ndarray, model, ST_map: np.ndarray = None, r: int = 3) -> np.ndarray:
    """
    Generates a probability map for the input image using the provided model.
    
    Args:
        - image: Input BGR image.
        - model: Trained model for prediction.
        - ST_map: Additional structure tensor value feature map.
        - r: Radius for local neighborhood in feature selection.

    Returns:
        np.ndarray: Probability map with same spatial dimensions as the input image.
    """
    if len(image.shape) == 2:
        image_gray = image
    else:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(image_gray, 50, 200)

    h, w = image_gray.shape
    prob_map = np.zeros((h, w), dtype=np.float32)

    for y in range(r, h - r):
        for x in range(r, w - r):
            # Optional: Skip non-edge pixels if needed
            # if edges[y, x] == 0:
            #     continue
            
            features = extract_features_at_pixel(image, x, y, r=r, ST_map=ST_map)
            if features is None:
                continue
            
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                prob = model(features_tensor).item()
            
            prob_map[y, x] = prob

    return prob_map

def find_ridge_from_prob_map_DP(prob_map: np.ndarray, delta: int = 5, epsilon: float = 1e-6) -> np.ndarray:
    """
    Find the most probable ridge/path through a 2D probability map using dynamic programming.

    Args:
        - prob_map: 2D array of probabilities, where higher values indicate more likely ridge positions.
        - delta: Maximum vertical shift allowed between adjacent columns in the ridge path.
        - epsilon: Small constant to avoid log(0) when computing cost.

    Returns:
        np.ndarray: 1D array of length W indicating the row index of the ridge in each column.
    """
    H, W = prob_map.shape
    cost = np.full((H, W), np.inf)
    backtrack = np.full((H, W), -1, dtype=int)

    # Convert prob_map to cost map using negative log
    log_cost = -np.log(prob_map + epsilon)

    # Initialize first column
    cost[:, 0] = log_cost[:, 0]

    for x in range(1, W):
        for y in range(H):
            y_min = max(y - delta, 0)
            y_max = min(y + delta + 1, H)
            for yy in range(y_min, y_max):
                transition_cost = abs(y - yy)
                total_cost = cost[yy, x-1] + transition_cost + log_cost[y, x]
                if total_cost < cost[y, x]:
                    cost[y, x] = total_cost
                    backtrack[y, x] = yy

    # Backtrace the path
    ridge = np.zeros(W, dtype=int)
    ridge[-1] = np.argmin(cost[:, -1])
    for x in range(W - 2, -1, -1):
        ridge[x] = backtrack[ridge[x + 1], x + 1]

    return ridge

def find_ridge_DP_Canny(image: np.ndarray, model, ST_map: np.ndarray = None, r: int = 3, delta: int = 5) -> np.ndarray:
    """
    Applies Canny edge detection, evaluates model on edge pixels to generate a sparse probability map,
    then finds the most probable ridge using dynamic programming.

    Args:
        - image: Input image.
        - model: Trained model to evaluate pixel level probability.
        - ST_map: Additional feature map for structure tensor values.
        - r: Neighborhood radius for feature extraction.
        - delta: Maximum vertical shift allowed between adjacent ridge points.

    Returns:
        np.ndarray: Ridge path as 1D array of row indices for each column.
    """
    if len(image.shape) == 2:
        image_gray = image
    else:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image_gray, 50, 200)
    h, w = image_gray.shape
    prob_map = np.zeros((h, w), dtype=np.float32)

    for y in range(r, h - r):
        for x in range(r, w - r):
            if edges[y, x] == 0:
                continue

            features = extract_features_at_pixel(image, x, y, r=r, ST_map=ST_map)
            if features is None:
                continue

            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prob = model(features_tensor).item()

            prob_map[y, x] = prob

    ridge = find_ridge_from_prob_map_DP(prob_map, delta=delta)

    return ridge

def find_ridge_greedy(image: np.ndarray, model, ST_map: np.ndarray = None, r: int = 3, search_r: int = 3, restart_threshold: float = 0.8) -> list:
    """
    Tracks a ridge across the image using a greedy approach based on model-predicted probabilities
    and extracted features at each pixel.

    Args:
        - image: Input BGR or grayscale image.
        - model: Trained model for prediction.
        - ST_map: Additional structure tensor value feature map.
        - r: Radius for local neighborhood in feature extraction.
        - search_r: Search radius for greedy algo
        - restart_threshold: Threshold below which to restart prediction on new vertical

    Returns:
        list: List of row indices representing the detected ridge across the image width.
    """
    h, w = image[:, :, 0].shape if image.ndim == 3 else image.shape
    ridge = []
    first_column = [0 for _ in range(r)]

    for y in range(r, h - r):
        features = extract_features_at_pixel(image, r, y, r=r, ST_map=ST_map)
        if features is None:
            continue
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prob = model(features_tensor).item()
        first_column.append(prob)

    first_column_sorted = sorted(first_column, reverse=True)
    pred_mtn_height = first_column.index(first_column_sorted[0])
    for i in range(r):
        ridge.append(pred_mtn_height)
    
    search_space = []
    ridge_probs = [1, 1, 1]
    x = r + 1
    t = 0


    while x < (w - r):
        for y in range(pred_mtn_height - search_r, pred_mtn_height + search_r + 1):
            features = extract_features_at_pixel(image, x, y, r=r, ST_map=ST_map)
            if features is None:
                continue
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prob = model(features_tensor).item()
            search_space.append(prob)
        
        pred_mtn_height -= r - search_space.index(max(search_space))
        pred_mtn_height = max(r+search_r, min(pred_mtn_height, h - r - search_r-1))

        ridge.append(pred_mtn_height)

        ridge_probs.append(max(search_space))

        x += 1
        search_space = []

    return ridge
