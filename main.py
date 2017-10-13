# Title: Detection and classification of moving objects based on the real-time camera image.
# ©WinczewskiDamian2017

import cv2

cap = cv2.VideoCapture('http://camera.buffalotrace.com/mjpg/video.mjpg?timestamp=1507887365324')

while True:
  ret, frame = cap.read()
  cv2.imshow('Video', frame)

  if cv2.waitKey(1) == 27:
    exit(0)
