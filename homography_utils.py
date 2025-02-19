import cv2
import numpy as np


def apply_homography_patch(frames, ref_patch, ref_coords, dst_coords):
    # ref_coords: (sx, sy, ex, ey) source patch region
    # dst_coords: (sx, sy, ex, ey) destination patch region
    frame_height, frame_width = frames[0].shape[:2]

    ref_sx, ref_sy, ref_ex, ref_ey = ref_coords
    dst_sx, dst_sy, dst_ex, dst_ey = dst_coords

    src_pts = np.array(
        [[ref_sx, ref_sy], [ref_ex, ref_sy], [ref_ex, ref_ey], [ref_sx, ref_ey]],
        dtype=np.float32,
    )
    dst_pts = np.array(
        [[dst_sx, dst_sy], [dst_ex, dst_sy], [dst_ex, dst_ey], [dst_sx, dst_ey]],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    warped_patch = cv2.warpPerspective(ref_patch, H, (frame_width, frame_height))
    patch_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.fillConvexPoly(patch_mask, np.int32(dst_pts), 255)

    return warped_patch, patch_mask
