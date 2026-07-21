# Finger Pattern Analysis System

A desktop application for analyzing finger exploration patterns on tactile graphics. The application provides an end-to-end pipeline for model training, salient region annotation, video analysis, and visualization of exploration patterns.

---

## Features

- Train YOLO models for tactile graphics
- Mark salient regions on tactile graphics
- Analyze participant video folders
- Generate cumulative analysis results
- Visualize exploration trajectories and analytical outputs
- Interactive desktop interface built with PySide6

---

## Prerequisites

Before running the application, ensure the following are installed:

- Python **3.10** or **3.11** *(Python 3.9 is also supported)*
- **pip** (comes with Python)
- Windows, macOS, or Linux

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Finger_Exploration_APP
```

Or download the repository as a ZIP file and extract it.

---

### 2. Install Dependencies

#### Windows

Upgrade pip

```bash
python -m pip install --upgrade pip
```

Install required packages

```bash
python -m pip install -r application\requirements.txt
```

If required, install the following packages manually:

```bash
python -m pip install mediapipe ultralytics opencv-python torch torchvision torchaudio PySide6
```

---

#### macOS / Linux

Upgrade pip

```bash
python3 -m pip install --upgrade pip
```

Install required packages

```bash
python3 -m pip install -r application/requirements.txt
```

If required, install the following packages manually:

```bash
python3 -m pip install mediapipe ultralytics opencv-python torch torchvision torchaudio PySide6
```

---

## Running the Application

### Windows

```bash
python application\app.py
```

### macOS / Linux

```bash
python3 application/app.py
```

---

## Project Workflow

The application follows a **4-step pipeline**:

### 1. Train YOLO Model

- Load the corresponding `data.yaml`
- Fine-tune the YOLO model
- Generate the trained `best.pt` model

### 2. Mark Salient Regions

- Load the tactile graphic
- Annotate salient regions
- Export annotations as `regions.json`

### 3. Analyze Video Folder

- Load:
  - Trained model (`best.pt`)
  - `regions.json`
  - Participant video folder
- Run cumulative analysis

### 4. Results

- Load generated results
- View trajectory plots
- Visualize analytical outputs
- Load summary JSON files

---

## Project Structure

```text
Finger_Exploration_APP/
│
├── application/
│   ├── app.py
│   ├── requirements.txt
│   ├── assets/
│   │   └── hand_landmarker.task
│   └── ...
│
├── README.md
└── ...
```

---

## Notes

- Ensure the file

```text
application/assets/hand_landmarker.task
```

is present before running the application.

If you encounter a `ModuleNotFoundError`, install the missing package:

**Windows**

```bash
python -m pip install <package_name>
```

**macOS / Linux**

```bash
python3 -m pip install <package_name>
```

---

## Tech Stack

- Python
- PySide6
- OpenCV
- MediaPipe
- Ultralytics YOLO
- PyTorch

## Current Capabilities

The system provides an end-to-end pipeline that detects objects, performs orientation correction, tracks finger movement, maps interactions to predefined regions, extracts behavioural sequences, and generates graph-based analytical visualisations from video data.

<img width="1918" height="1078" alt="Screenshot 2026-06-15 222411" src="https://github.com/user-attachments/assets/4d268e6a-5cca-486b-87a1-42e255044019" />
<img width="1918" height="1078" alt="Screenshot 2026-06-15 222421" src="https://github.com/user-attachments/assets/d1226285-4b69-40e4-95e5-a6ecb73489c0" />
<img width="1918" height="1078" alt="Screenshot 2026-06-15 222429" src="https://github.com/user-attachments/assets/2ac0d410-5f17-495e-9cc9-96a2486d7d5d" />
<img width="1918" height="1078" alt="Screenshot 2026-06-15 222436" src="https://github.com/user-attachments/assets/afe80bb0-200d-4554-b9e2-6fcc6a22cfdb" />

---

## License

This project is intended for academic and research purposes.



