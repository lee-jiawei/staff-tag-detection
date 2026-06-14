import cv2
from pathlib import Path


def extract_frames(video_path: str = "video/sample.mp4", output_dir: str = "raw_frames", samples_per_sec: int = 3) -> None:
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fps = video.get(cv2.CAP_PROP_FPS)
    interval = max(int(fps / samples_per_sec), 1)

    frame_count = 0
    image_count = 0

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if frame_count % interval == 0:
            output_path = Path(output_dir) / f"{image_count}.jpg"
            cv2.imwrite(str(output_path), frame)
            image_count += 1
        frame_count += 1

    video.release()
    cv2.destroyAllWindows()
    print(f"Saved {image_count} frames to {output_dir}")


if __name__ == "__main__":
    extract_frames()
