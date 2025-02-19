from model_loader import load_pipeline
from marker_detection import detect_all_markers
from video_io import load_video_frames, save_video_frames
from marker_processing import process_marker
from constants import INPUT_VIDEO, OUTPUT_VIDEO


def run_pipeline():
    # Load model
    pipe, generator, device = load_pipeline()

    # Load video frames
    frames, fps, frame_width, frame_height = load_video_frames(INPUT_VIDEO)

    # Detect markers
    all_markers, unique_ids = detect_all_markers(frames)
    print("Found marker IDs:", unique_ids)

    # Process each marker
    for marker_id in unique_ids:
        print(f"Processing Marker ID {marker_id}...")
        frames = process_marker(
            marker_id,
            frames,
            all_markers,
            pipe,
            generator,
            device,
            frame_width,
            frame_height,
        )
        print(f"Marker {marker_id} processing done.")

    # Save the processed video
    save_video_frames(frames, OUTPUT_VIDEO, fps, frame_width, frame_height)
    print("All markers processed. Output saved to:", OUTPUT_VIDEO)
