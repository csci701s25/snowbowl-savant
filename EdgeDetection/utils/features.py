import numpy as np

def extract_features_at_pixel(image: np.ndarray,x: int, y: int, r: int, ST_map: np.ndarray = None,) -> np.ndarray:
    """
    Extracts features (surrounding patch and structure tensor) from an image coordinate.

    Options:
    - image: input image
    - x: x image coordinate
    - y: y image coordinate
    - r: patch radius
    - ST_map: If False, return only input image path
              If exists, add ST values to feature list


    Returns:
    - A 1D feature vector combining pixel values and optional ST features
    """
    patch = image[y - r:y + r + 1, x - r:x + r + 1].astype(np.float32) / 255.0
    patch_features = patch.ravel()

    if ST_map is not None:
        S, C, O = ST_map
        st_features = np.array([S[y, x], C[y, x], O[y, x]], dtype=np.float32)
        return np.concatenate([patch_features, st_features])
    else:
        return patch_features
