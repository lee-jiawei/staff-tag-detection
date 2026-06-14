# Staff Tag Detection
Detect and track person wearing a piano tag as staff in an office.

## Installation

Clone this repository or download as zip file:

```bash
git clone https://github.com/
```

Install dependencies

```bash
pip install -r requirements.txt
```

## Repository Structure

- `main.ipynb` - jupyter notebook for frame extraction, dataset download, training, inference
- `scripts/extract_frames.py` - frames from input video for manual annotation in Roboflow
- `scripts/train_tag_model.py` - fine tune yolov8n model with annotated piano tag images 
- `scripts/run_inference.py` - run detection, tracking, association, and export outputs
- `config.yaml` - default paths and thresholds
- `staff-tag-detection-1/` - Roboflow YOLO dataset export
- `output/output.mp4` - annotated output video (generated)
- `output/staff.csv` - timestamp and coordinates of detected staff (generated)

## Results
Demo video 
- [Youtube Link](https://youtu.be/VM4Jzv85wOc)

Annotated video
- `output/output.mp4`

Staff details
- `output/staff.csv`

##  Execution

```bash
python3 scripts/extract_frames.py
python3 scripts/train_tag_model.py
python3 scripts/run_inference.py
```
