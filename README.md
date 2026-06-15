# Tactile Graphic Interaction Analysis System

A computer vision-based application for analyzing tactile graphic interaction patterns from videos using YOLOv8 OBB detection, MediaPipe finger tracking, spatial-temporal modeling, and graph-based behavioral analysis.

## Features

* Upload custom datasets using `data.yaml`
* Train YOLOv8 object detection models directly through the GUI
* Perform inference on images and videos
* Detect objects using Oriented Bounding Boxes (OBB)
* Apply perspective transformation and image warping
* Track fingertip movements using MediaPipe Hand Landmarks
* Define salient interaction regions (A, B, C, D, E)
* Generate cumulative finger trajectory visualizations
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
* Human behavior and interaction studies
* Spatial-temporal movement analysis
* Computer vision research
* Educational and accessibility research

## Current Capabilities

The system provides an end-to-end pipeline that detects objects, performs orientation correction, tracks finger movement, maps interactions to predefined regions, extracts behavioral sequences, and generates graph-based analytical visualizations from video data.
