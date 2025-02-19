import cv2


def load_video_frames(input_video):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise IOError("Error: Could not open video.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25

    frames = []
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    return frames, fps, frame_width, frame_height


def save_video_frames(frames, output_video, fps, frame_width, frame_height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    for f in frames:
        out.write(f)
    out.release()
