# Title: Detection and classification of moving objects based on the real-time camera image.
# ©WinczewskiDamian2017

import numpy as np
import cv2
import sys


def system_info():
    print("The Python version is %s.%s.%s" % sys.version_info[:3])
    print("The OpenCV version is", cv2.__version__)


def capture_video():
    cap = cv2.VideoCapture('http://camera.buffalotrace.com/mjpg/video.mjpg?timestamp=1507887365324')
    # cap = cv2.VideoCapture(0) #camera systemowa
    print("Frame default resolution: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; "
          + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")"
          + "Frame rate: " + str(cap.get(cv2.CAP_PROP_FPS)))
    return cap


def show_video(cap):
    history = 2000
    detectShadows = False
    varThreshold = 16   # default 16
    fgbg = cv2.createBackgroundSubtractorMOG2(history=history,
                                              detectShadows=detectShadows,
                                              varThreshold=varThreshold)
    dist2Threshold = 400.0  # default 400.0
    knnbg = cv2.createBackgroundSubtractorKNN(history=history,
                                              detectShadows=detectShadows,
                                              dist2Threshold=dist2Threshold)

    while True:
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        knnmask = knnbg.apply(frame)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Video', gray)
        cv2.imshow('Mask KNN', knnmask)
        cv2.imshow('Mask MOG2', fgmask)

        # cascPath = sys.argv[2]
        # faceCascade = cv2.CascadeClassifier(cascPath)
        # # print(ret)
        #
        # faces = faceCascade.detectMultiScale(
        #     gray,
        #     scaleFactor=1.1,
        #     minNeighbors=5,
        #     minSize=(30, 30),
        #     flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        # )

        if cv2.waitKey(1) == 27:
            # print("Found {0} faces!".format(len(faces)))
            print("[INFO] cleaning up...")
            cv2.destroyAllWindows()
            print('x')
            break
            # exit(0)
            # if cv2.getWindowProperty('Camera', 0) == -1:
            #     print('x')
            #     break


if __name__ == '__main__':
    system_info()
    cap = capture_video()
    show_video(cap)
