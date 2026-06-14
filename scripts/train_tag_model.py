from ultralytics import YOLO


def train_tag_model(
    base_weights: str = "yolov8n.pt",
    data_yaml: str = "staff-tag-detection-1/data.yaml",
    epochs: int = 50,
    imgsz: int = 640,
    device: str = "cpu",
) -> None:
    model = YOLO(base_weights)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, device=device)


if __name__ == "__main__":
    train_tag_model()
