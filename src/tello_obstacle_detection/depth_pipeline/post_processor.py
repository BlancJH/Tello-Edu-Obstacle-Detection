import numpy as np

def normalize_midas_depth(depth_map: np.ndarray) -> np.ndarray:
    """
    Normalize MiDaS depth output to range [0, 1].
    
    Args:
        depth_map (np.ndarray): Raw depth prediction from MiDaS.
    
    Returns:
        np.ndarray: Normalized depth map in [0, 1].
    """
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    
    # Avoid division by zero if depth is uniform
    if max_val - min_val < 1e-6:
        return np.zeros_like(depth_map)
    
    normalized = (depth_map - min_val) / (max_val - min_val)
    return normalized

def safe_direction(depth_norm: np.ndarray, near_thresh: float=0.35, min_free: float=0.65) -> str:
    """
    Decide the safest forward direction (center > right > left) from a normalized MiDaS depth map.

    The function analyzes a central horizontal band of the depth map, splits it into
    left/center/right sectors, and determines which sector has sufficient free space.
    "Free space" is defined as pixels with normalized depth below the near_thresh
    (i.e., farther away than the threshold). The function returns the first safe
    direction in priority order: center → right → left. If none are safe, "none" is returned.

    Args:
        depth_norm (np.ndarray): Normalized depth map in [0,1], larger values ≈ closer obstacles.
        near_thresh (float, optional): Threshold above which pixels are considered obstacles. Default is 0.35.
        min_free (float, optional): Minimum fraction of free pixels required in a sector to deem it safe. Default is 0.65.

    Returns:
        str: One of {"center", "right", "left", "none"} indicating the safest direction.
    """
    h, w = depth_norm.shape
    y0, y1 = int(h*0.35), int(h*0.65)  # central forward band for drones
    band = depth_norm[y0:y1, :]
    blocked = band >= near_thresh  # MiDaS: larger ~ closer
    free = ~blocked
    thirds = [(0, w//3), (w//3, 2*w//3), (2*w//3, w)]
    names = ["left","center","right"]
    free_ratios = {}
    for name, (x0, x1) in zip(names, thirds):
        sector = free[:, x0:x1]
        free_ratios[name] = float(sector.mean()) if sector.size else 0.0
    # Priority: center > right > left
    if free_ratios["center"] >= min_free:
        return "center"
    elif free_ratios["right"] >= min_free:
        return "right"
    elif free_ratios["left"] >= min_free:
        return "left"
    else:
        return "none"