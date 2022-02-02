# Interview test for security robotics
Applicant & author: Ruoyu Peng (ruoyu.peng123@gmail.com)

## Introduction of the software
TBD

## Dependencies
- Tensorflow 2.7.0 
- Gstreamer 1.0 (https://gstreamer.freedesktop.org/documentation/index.html?gi-language=python)
- TensorFlow Object Detection API (https://github.com/tensorflow/models/tree/master/research/object_detection). 
  Installation instruction: (https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) 

## Usage

To start the software:
- python rtsp_od.py
  open VLC, click Media -> Open Network Stream
- type the address rtsp://127.0.0.1:8554/test in the URL
- click play

By default, the video source is "target_video.mp4", which contains persons and cars on the street.

To use the webcam instead, change in the source file 

cap = cv2.VideoCapture("target_video.mp4") to cap = cv2.VideoCapture(0)

