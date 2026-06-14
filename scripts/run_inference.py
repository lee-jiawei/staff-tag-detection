import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path


def run_inference(
    input_video: str = "video/sample.mp4",
    tag_model_path: str = "runs/detect/train/weights/best.pt",
    person_model_path: str = "best.pt",
    tracker_cfg: str = "bytetrack_staff.yaml",
    output_video: str = "output/output.mp4",
    output_csv: str = "output/staff.csv",
    person_conf_thresh: float = 0.2,
    tag_conf_thresh: float = 0.2,
    staff_min_frames: int = 5,
) -> None:
    tag_model = YOLO(tag_model_path)
    person_model = YOLO(person_model_path)

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(input_video)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video file: {input_video}")

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        video.get(cv2.CAP_PROP_FPS),
        (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )

    staff_set = set()
    staff_frame_count = {}
    staff_dict = {}
    frame_count = 0
    records = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1
        person_results = person_model.track(frame, persist=True, tracker=tracker_cfg, conf=person_conf_thresh)
        tag_results = tag_model.track(frame, persist=True)

        for tag in tag_results[0].boxes:
            tx1, ty1, tx2, ty2 = map(int, tag.xyxy[0])
            tag_center_x = (tx1 + tx2) / 2
            tag_center_y = (ty1 + ty2) / 2
            conf_tag = float(tag.conf[0])

            if conf_tag <= tag_conf_thresh:
                continue

            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
            cv2.putText(frame, f"Tag {conf_tag:.2f}", (tx1, max(ty1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            for person in person_results[0].boxes:
                if person.id is None:
                    continue
                px1, py1, px2, py2 = map(int, person.xyxy[0])
                person_id = int(person.id[0])
                conf_person = float(person.conf[0])

                if conf_person >= person_conf_thresh and (px1 < tag_center_x < px2 and py1 < tag_center_y < py2):
                    staff_frame_count[person_id] = staff_frame_count.get(person_id, 0) + 1
                    if staff_frame_count[person_id] > staff_min_frames:
                        staff_set.add(person_id)
            break

        fps = video.get(cv2.CAP_PROP_FPS)
        total_seconds = frame_count / fps if fps else 0
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)

        for person in person_results[0].boxes:
            if person.id is None:
                continue
            person_id = int(person.id[0])
            if person_id not in staff_set:
                continue

            px1, py1, px2, py2 = map(int, person.xyxy[0])
            conf_person = float(person.conf[0])
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(frame, f"Staff {conf_person:.2f}", (px1, max(py1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            staff_x = int((px1 + px2) / 2)
            staff_y = int((py1 + py2) / 2)
            staff_info = {
                "timestamp": f"{minutes:02d}:{seconds:02d}",
                "x": staff_x,
                "y": staff_y,
                "person_conf": round(conf_person, 2),
            }
            staff_dict[person_id] = staff_info
            records.append(staff_info)

        if staff_dict:
            latest = staff_dict[list(staff_dict.keys())[-1]]
            cv2.rectangle(frame, (10, 5), (230, 35), (255, 255, 255), -1)
            cv2.putText(frame, f"Coordinate(x,y): {latest['x']},{latest['y']}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        out.write(frame)
        cv2.imshow("Staff Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pd.DataFrame(records).to_csv(output_csv, index=False)
    video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference()
