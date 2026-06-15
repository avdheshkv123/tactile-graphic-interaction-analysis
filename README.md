# Tactile Graphic Interaction Analysis System

A computer vision-based application for analysing tactile graphic interaction patterns from videos using YOLOv8 OBB detection, MediaPipe finger tracking, spatial-temporal modelling, and graph-based behavioural analysis.

## Features

* Upload custom datasets using `data.yaml`
* Train YOLOv8 object detection models directly through the GUI
* Perform inference on images and videos
* Detect objects using Oriented Bounding Boxes (OBB)
* Apply perspective transformation and image warping
* Track fingertip movements using MediaPipe Hand Landmarks
* Define salient interaction regions (A, B, C, D, E)
* Generate cumulative finger trajectory visualisations
* Extract region visit sequences and transition patterns
* Create interaction frequency graphs and movement direction plots
* Export analysis results and structured JSON outputs

## Workflow

Dataset Upload → Model Training → Object Detection → Perspective Warp → Finger Tracking → Region Mapping → Sequence Extraction → Graph Analysis

## Technologies Used

* Python
* YOLOv8
* MediaPipe
* OpenCV
* NumPy
* Pandas
* Matplotlib
* PyQt/Tkinter
* Google Colab
* Kaggle
* Roboflow

## Applications

* Tactile graphic interaction analysis
* Human behaviour and interaction studies
* Spatial-temporal movement analysis
* Computer vision research
* Educational and accessibility research

## Current Capabilities

The system provides an end-to-end pipeline that detects objects, performs orientation correction, tracks finger movement, maps interactions to predefined regions, extracts behavioural sequences, and generates graph-based analytical visualisations from video data.

<img width="1918" height="1078" alt="Screenshot 2026-06-15 222411" src="https://github.com/user-attachments/assets/4d268e6a-5cca-486b-87a1-42e255044019" />
<img width="1918" height="1078" alt="Screenshot 2026-06-15 222421" src="https://github.com/user-attachments/assets/d1226285-4b69-40e4-95e5-a6ecb73489c0" />
<img width="1918" height="1078" alt="Screenshot 2026-06-15 222429" src="https://github.com/user-attachments/assets/2ac0d410-5f17-495e-9cc9-96a2486d7d5d" />
<img width="1918" height="1078" alt="Screenshot 2026-06-15 222436" src="https://github.com/user-attachments/assets/afe80bb0-200d-4554-b9e2-6fcc6a22cfdb" />



