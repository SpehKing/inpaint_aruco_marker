import cv2
import numpy as np
from constants import EXPANSION_FACTOR

# Initialize global detector
detector_params = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DETECTOR = cv2.aruco.ArucoDetector(aruco_dict, detector_params)


def detect_all_markers(frames):
    all_markers = []
    unique_ids = set()
    for f in frames:
        marker_corners, marker_ids, _ = DETECTOR.detectMarkers(f)
        if marker_ids is not None and len(marker_ids) > 0:
            marker_ids = marker_ids.flatten()
            marker_list = [
                (marker_corners[i], marker_ids[i]) for i in range(len(marker_ids))
            ]
            all_markers.append(marker_list)
            for mid in marker_ids:
                unique_ids.add(mid)
        else:
            all_markers.append([])
    return all_markers, sorted(list(unique_ids))


def create_mask_for_marker(image, corners, expansion_factor=EXPANSION_FACTOR):
    cx = np.mean(corners[:, 0])
    cy = np.mean(corners[:, 1])

    expanded_corners = []
    for x, y in corners:
        dx = x - cx
        dy = y - cy
        x_expanded = cx + dx * (1 + expansion_factor)
        y_expanded = cy + dy * (1 + expansion_factor)
        expanded_corners.append([x_expanded, y_expanded])

    expanded_corners = np.array(expanded_corners, dtype=np.int32)
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)
    cv2.fillPoly(mask, [expanded_corners], 1.0)
    return mask, (cx, cy)
