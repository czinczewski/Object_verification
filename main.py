# Title: Detection and classification of moving objects based on the real-time camera image.
# ©WinczewskiDamian2017

import numpy as np
import cv2
import sys


def system_info(cv2_info):
    print("The Python version is %s.%s.%s" % sys.version_info[:3])
    print("The OpenCV version is", cv2.__version__)
    if cv2_info:
        print("Cv2 build information", cv2.getBuildInformation())


def capture_video():
    cap = cv2.VideoCapture('http://camera.buffalotrace.com/mjpg/video.mjpg?timestamp=1507887365324')
    # cap = cv2.VideoCapture(0) #camera systemowa
    print("Frame default resolution: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; "
          + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")"
          + "Frame rate: " + str(cap.get(cv2.CAP_PROP_FPS)))
    return cap


def show_video(cap, grayscal, backgroundsub):
    if backgroundsub[0] or backgroundsub[1]:
        history = 200
        detectShadows = False
        if backgroundsub[0]:
            varThreshold = 16   # default 16
            fgbg = cv2.createBackgroundSubtractorMOG2(history=history,
                                                      detectShadows=detectShadows,
                                                      varThreshold=varThreshold)
        if backgroundsub[1]:
            dist2Threshold = 400.0  # default 400.0
            knnbg = cv2.createBackgroundSubtractorKNN(history=history,
                                                      detectShadows=detectShadows,
                                                      dist2Threshold=dist2Threshold)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Original', frame)

        if grayscal:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Video', gray)

        # frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

        if backgroundsub[0]:
            fgmask = fgbg.apply(frame)
            cv2.imshow('Mask MOG2', fgmask)

        if backgroundsub[1]:
            knnmask = knnbg.apply(frame)
            cv2.imshow('Mask KNN', knnmask)


        if cv2.waitKey(1) == 27:
            # print("Found {0} faces!".format(len(faces)))
            print("[INFO] cleaning up...")
            cv2.destroyAllWindows()
            print('x')
            break
            # exit(0)
            if cv2.getWindowProperty('Original', 0) == -1:
                cv2.destroyAllWindows()
                print('x')
                break


if __name__ == '__main__':
    system_info(False)
    backgroundsub = [False, False]
    grayscal = False
    cap = capture_video()
    show_video(cap, grayscal, backgroundsub)
