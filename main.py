# Title: Detection and classification of moving objects based on the real-time camera image.
# ©WinczewskiDamian2017

import numpy as np
import cv2
import sys


# body_cascade = cv2.CascadeClassifier('./Haarcascade_body/deuflat_from_opencv/haarcascade_frontalface_default.xml')
haarcascade_upperbody = cv2.CascadeClassifier('./Haarcascade_body/HS.xml')
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def system_info(cv2_info):
    print("The Python version is %s.%s.%s" % sys.version_info[:3])
    print("The OpenCV version is", cv2.__version__)
    if cv2_info:
        print("Cv2 build information", cv2.getBuildInformation())

    help(cv2.CascadeClassifier().detectMultiScale)


def capture_video(my_camera):
    if my_camera:
        cap = cv2.VideoCapture(0) #camera systemowa
    else:
        cap = cv2.VideoCapture('http://camera.buffalotrace.com/mjpg/video.mjpg?timestamp=1507887365324')

    print("Frame default resolution: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; "
          + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")"
          + "Frame rate: " + str(cap.get(cv2.CAP_PROP_FPS)))
    return cap


def detect(gray, frame):
    body = haarcascade_upperbody.detectMultiScale(gray,
                                                    scaleFactor=1.05, minNeighbors=2,
                                                    flags=0,
                                                    minSize=(20, 20), maxSize=(150, 150))
    body2 = haarcascade_upperbody.detectMultiScale(gray,
                                                    scaleFactor=1.1, minNeighbors=2,
                                                    flags=0,
                                                    minSize=(20, 20), maxSize=(150, 150))

    # (bboxes, confidences) = hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

    print("Bodies Up:", len(body), "Bodies Full:", len(body2))
    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    for (x, y, w, h) in body2:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame


def show_video(cap, original, grayscal, backgroundsub):
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
        _, frame = cap.read()
        # frame = cv2.GaussianBlur(frame, (3, 3), 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.fastNlMeansDenoisingMulti(gray, 2, 5, None, 4, 7, 35)
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if original:
            cv2.imshow('Original', frame)

        if grayscal:
            cv2.imshow('Video', gray)

        # frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

        if backgroundsub[0]:
            fgmask = fgbg.apply(frame)
            cv2.imshow('Mask MOG2', fgmask)

        if backgroundsub[1]:
            knnmask = knnbg.apply(frame)
            cv2.imshow('Mask KNN', knnmask)

        frame = detect(gray, frame)
        cv2.imshow("Output", frame)


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
    original = False
    cap = capture_video(False)
    show_video(cap, original, grayscal, backgroundsub)
