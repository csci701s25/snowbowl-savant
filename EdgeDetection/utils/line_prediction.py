import numpy as np
import cv2
import torch
from utils.features import extract_features_at_pixel

def generate_probability_map(image: np.ndarray, model, ST_map: np.ndarray = None, r: int = 3) -> np.ndarray:
    """
    Generates a probability map for the input image using the provided model.
    
    Args:
        image (np.ndarray): Input BGR image.
        model (torch.nn.Module): Trained model for prediction.
        ST_map (np.ndarray, optional): Additional feature map. Defaults to None.
        r (int): Radius for local neighborhood. Defaults to 3.

    Returns:
        np.ndarray: Probability map with same spatial dimensions as the input image.
    """
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
