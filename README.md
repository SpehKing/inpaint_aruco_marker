# ArUco Marker Removal from Video Using Diffusion-Based Inpainting

## Overview

This project is designed to remove ArUco markers from video frames using a two-stage process:

1. **Diffusion-Based Inpainting:** A Stable Diffusion inpainting model fills in the masked regions where markers are detected.
2. **Temporal Consistency & Homography:** A SIFT-based approach selects the best inpainted candidate frame, and homography is used to seamlessly paste the patch onto other frames.

The end result is a video with the ArUco markers removed and the inpainted content blended across frames for visual consistency.

**Demo Example:**

- **Input Video:** `/video/demo_input.mp4`
- **Output Video:** `/video/demo_output.gif`

_Note:_ Although the primary supported input format is MP4, the demo output is shown as a GIF for quick visualization. Adjust your processing settings accordingly if you need a different output format.

## Features

- **Video Input/Output:**
  - **Input format:** MP4
  - **Output format:** MP4 (or GIF for demo purposes)
  - **Frame rate:** 25 FPS (input and output)
- **ArUco Marker Detection:**
  - Utilizes OpenCV’s ArUco module with the `DICT_6X6_250` dictionary.
- **Inpainting Pipeline:**
  - Uses the `StableDiffusionInpaintPipeline` (via the diffusers library) for image inpainting.
  - Inpainting parameters:
    - Empty text prompt.
    - Inpainting strength set to 0.99.
- **Mask Generation:**
  - Masks are generated as polygonal regions based on the precise marker corner coordinates.
- **Temporal Consistency:**
  - Uses SIFT keypoints to evaluate multiple candidate inpainted patches.
  - The candidate patch with the lowest keypoint count (indicative of a smoother inpaint) is selected.
- **Patch Application via Homography:**
  - Computes homography using `cv2.findHomography` with RANSAC.
  - Applies a perspective warp to the best candidate patch and blends it into each frame.

## Architecture & Pipeline

1. **Video Decoding:**
   - Read the input MP4 video ensuring it conforms to 25 FPS.
2. **ArUco Marker Detection:**
   - Detect markers in each frame using `cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)`.
3. **Mask Creation:**
   - Generate a polygon mask using the exact corner coordinates of each detected marker.
4. **Inpainting:**
   - For each detected marker region, run the diffusion-based inpainting process with an empty text prompt and a strength of 0.99.
5. **Candidate Patch Selection (SIFT-based):**
   - For each candidate patch from different frames, compute SIFT keypoints.
   - The patch with the lowest keypoint count is considered the best candidate.
6. **Homography and Patch Application:**
   - Compute the homography between the candidate patch and target frame regions using `cv2.findHomography` with RANSAC.
   - Warp and blend the best patch into the corresponding region in each frame.
7. **Video Encoding:**
   - Combine the processed frames back into an MP4 video at 25 FPS (or output as a GIF for demo purposes).

## Installation & Requirements

### Python Version

- **Python 3.12.8**

### Dependencies

The following Python libraries are required:

- **torch** (for GPU support and tensor computations)
- **diffusers** (for Stable Diffusion inpainting)
- **opencv-contrib-python** (for ArUco detection and SIFT)
- **numpy** (for numerical operations)
- **transformers** (typically required by diffusers)

A sample `requirements.txt` is provided below.

#### requirements.txt

```txt
# Python version: 3.12.8
torch>=2.0.1
diffusers>=0.20.0
opencv-contrib-python>=4.8.0.76
numpy>=1.23.0
transformers>=4.30.0
```

### Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-repo/aruco-marker-removal.git
   cd aruco-marker-removal
   ```

2. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download/Setup Model Weights:**

   - The inpainting model is loaded from Hugging Face (`benjamin-paine/stable-diffusion-v1-5-inpainting`). Ensure you have the necessary access if required.

5. **Run the Application:**
   - The main script handles video reading, processing, and writing. For example:
     ```bash
     python main.py --input_video /video/demo_input.mp4 --output_video /video/demo_output.gif
     ```
