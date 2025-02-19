import numpy as np
import cv2
from marker_detection import create_mask_for_marker
from inpainting import inpaint_region
from homography_utils import apply_homography_patch
from crop_utils import safe_crop_center
from constants import (
    CROP_SIZE,
    GUIDANCE_SCALE,
    NUM_INFERENCE_STEPS,
    STRENGTH,
    CROP_PERCENTAGE,
)


def find_nearest_frame_with_patch(current_index, available_indices):
    if not available_indices:
        return None
    differences = [abs(current_index - idx) for idx in available_indices]
    min_diff = min(differences)
    nearest_idx = available_indices[differences.index(min_diff)]
    return nearest_idx


def process_marker(
    marker_id, frames, all_markers, pipe, generator, device, frame_width, frame_height
):
    # Extract frames where this marker appears
    frames_with_marker = [
        i
        for i, fm in enumerate(all_markers)
        if any(m_id == marker_id for _, m_id in fm)
    ]
    if not frames_with_marker:
        return frames

    # Select candidate frames: only every 10th frame containing the marker
    candidate_frames = frames_with_marker[::10]

    # Dictionary to store patches for candidate frames only
    candidate_patches = {}

    # Inpaint patches for candidate frames
    for i in candidate_frames:
        frame_markers = all_markers[i]
        marker_data = [(c, mid) for (c, mid) in frame_markers if mid == marker_id]
        if not marker_data:
            continue

        c, _ = marker_data[0]
        c_reshaped = c.reshape(4, 2)
        marker_mask, (cx, cy) = create_mask_for_marker(frames[i], c_reshaped)

        sx, sy, ex, ey, w, h = safe_crop_center(
            cx, cy, CROP_SIZE, frame_width, frame_height
        )
        cropped_img = frames[i][sy:ey, sx:ex].copy()
        cropped_mask = marker_mask[sy:ey, sx:ex].copy().astype(np.uint8) * 255

        # Inpaint the region for this candidate frame
        inpainted_region = inpaint_region(
            pipe,
            generator,
            cropped_img,
            cropped_mask,
            GUIDANCE_SCALE,
            NUM_INFERENCE_STEPS,
            STRENGTH,
        )

        candidate_patches[i] = {
            "patch": inpainted_region,
            "corners": c_reshaped,
            "center": (cx, cy),
            "crop_coords": (sx, sy, ex, ey),
        }

    # If no candidate patches were created, nothing to do
    if not candidate_patches:
        return frames

    # Compute SIFT keypoints for each candidate's inpainted patch
    sift = cv2.SIFT_create(contrastThreshold=0.1, sigma=2)
    frame_keypoint_counts = {}
    for fidx, data in candidate_patches.items():
        patch = data["patch"]
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray_patch, None)
        kp_count = len(kp) if kp is not None else 0
        frame_keypoint_counts[fidx] = kp_count

    # Find the best candidate (lowest keypoint count)
    best_candidate_frame = min(frame_keypoint_counts, key=frame_keypoint_counts.get)
    best_data = candidate_patches[best_candidate_frame]
    sx, sy, ex, ey = best_data["crop_coords"]
    best_patch = best_data["patch"]
    best_corners_global = best_data["corners"]
    best_sx, best_sy, best_ex, best_ey = best_data["crop_coords"]

    # Apply the best candidate patch directly to the best candidate frame
    frames[best_candidate_frame][sy:ey, sx:ex] = best_patch

    # For all other frames that contain the marker (including those that are not candidates),
    # use the homography from the best candidate frame to warp its patch onto them.
    for i in frames_with_marker:
        if i == best_candidate_frame:
            continue

        current_markers = all_markers[i]
        cur_corners = None
        for cset, mid in current_markers:
            if mid == marker_id:
                cur_corners = cset.reshape(4, 2)
                break
        if cur_corners is None:
            continue

        # Compute homography from best candidate corners to current corners
        best_corners_local = best_corners_global - np.array([best_sx, best_sy])
        H, _ = cv2.findHomography(best_corners_local, cur_corners, cv2.RANSAC, 5.0)
        frame_h, frame_w = frames[0].shape[:2]

        # Warp the best patch using the computed homography
        warped_patch = cv2.warpPerspective(best_patch, H, (frame_w, frame_h))

        # Create a polygon mask based on the current corners, scaled outward
        centroid = np.mean(cur_corners, axis=0)
        scaling_factor = 1.8  # Adjust as needed
        scaled_polygon = centroid + scaling_factor * (cur_corners - centroid)
        polygon = scaled_polygon.astype(np.int32)

        # Create a mask for the polygon region in the current frame
        polygon_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [polygon], 255)

        # Blend the warped patch into the current frame using the polygon mask
        mask = polygon_mask == 255
        frames[i][mask] = warped_patch[mask]

    return frames
