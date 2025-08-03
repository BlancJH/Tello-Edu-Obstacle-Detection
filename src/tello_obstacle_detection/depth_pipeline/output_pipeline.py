import threading
import time
from typing import Callable, Optional
import requests
import numpy as np
import io

# Configuration
DEPTH_SERVER_URL = "http://localhost:8000/depth"  # adjust if your MIDAS exposes another path
FETCH_INTERVAL = 0.2  # seconds between automatic fetch attempts
MAX_DEPTH_AGE = 1.0  # seconds: consider depth stale after this
HTTP_TIMEOUT = 1.0  # seconds for each request
RETRY_BACKOFF = 0.5  # seconds to wait on failure before retrying

# Shared state
_lock = threading.Lock()
_latest_depth: Optional[np.ndarray] = None
_latest_timestamp: float = 0.0
_latest_meta: dict = {}  # can hold e.g., {"heading": ..., "frame_id": ...}

# Subscribers who want to be notified when new depth arrives
_subscribers: list[Callable[[np.ndarray, dict], None]] = []

def fetch_depth_once():
    """
    Attempts to fetch a depth map from the MIDAS depth endpoint.
    Expects the endpoint to return a .npy blob (NumPy array) or JSON with base64 if adapted.
    Updates internal cache on success.
    """
    global _latest_depth, _latest_timestamp, _latest_meta
    try:
        resp = requests.get(DEPTH_SERVER_URL, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        # Assume raw .npy content
        buf = io.BytesIO(resp.content)
        depth_map = np.load(buf, allow_pickle=False)

        with _lock:
            _latest_depth = depth_map
            _latest_timestamp = time.time()
            # Optionally, you could parse headers for metadata or include in a companion endpoint
            _latest_meta = {"fetched_at": _latest_timestamp}
        # notify subscribers
        for cb in list(_subscribers):
            try:
                cb(depth_map, _latest_meta)
            except Exception:
                pass  # subscriber errors shouldn't break fetcher
        return True
    except Exception as e:
        # Could log the error in real system
        return False

def background_fetcher(stop_event: threading.Event):
    """
    Runs in background, periodically pulling depth maps.
    """
    while not stop_event.is_set():
        success = fetch_depth_once()
        if not success:
            time.sleep(RETRY_BACKOFF)
        else:
            time.sleep(FETCH_INTERVAL)

def get_latest_depth(max_age: float = MAX_DEPTH_AGE) -> Optional[np.ndarray]:
    """
    Returns the latest depth map if it is fresh enough; otherwise None.
    """
    with _lock:
        if _latest_depth is None:
            return None
        age = time.time() - _latest_timestamp
        if age > max_age:
            return None
        return _latest_depth.copy()

def get_depth_age() -> float:
    with _lock:
        return time.time() - _latest_timestamp if _latest_depth is not None else float("inf")

def subscribe(callback: Callable[[np.ndarray, dict], None]):
    """
    Register a callback to be invoked whenever a new depth map is fetched.
    Callback signature: (depth_map: np.ndarray, meta: dict)
    """
    _subscribers.append(callback)

def unsubscribe(callback: Callable[[np.ndarray, dict], None]):
    try:
        _subscribers.remove(callback)
    except ValueError:
        pass

# Helper to start the fetcher thread
def start_background_fetcher() -> threading.Event:
    """
    Starts the background fetcher thread. Returns a stop_event you can set to terminate it.
    """
    stop_event = threading.Event()
    thread = threading.Thread(target=background_fetcher, args=(stop_event,), daemon=True)
    thread.start()
    return stop_event