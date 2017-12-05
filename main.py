# Title: Detection and classification of moving objects based on the real-time camera image.
# ©WinczewskiDamian2017

import numpy as np
import cv2
import sys


# body_cascade = cv2.CascadeClassifier('./Haarcascade_body/deuflat_from_opencv/haarcascade_frontalface_default.xml')
haarcascade_head_and_shoulders = cv2.CascadeClassifier('./Haarcascade_body/another/cascadeH5.xml')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def system_info(cv2_info):
    print("The Python version is %s.%s.%s" % sys.version_info[:3])
    print("The OpenCV version is", cv2.__version__)
    if cv2_info:
        print("Cv2 build information", cv2.getBuildInformation())

    # help(cv2.CascadeClassifier().detectMultiScale)


def capture_video(my_camera):
    if my_camera:
        # cap = cv2.VideoCapture(0) #camera systemowa
        # cap = cv2.VideoCapture('./Videos/2017_11_27_T_12_47_28_output.avi')
        cap = cv2.VideoCapture('./Videos/dev_stream.avi')
    else:
        cap = cv2.VideoCapture('http://camera.buffalotrace.com/mjpg/video.mjpg?timestamp=1507887365324')


    print("Frame default resolution: (" + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; "
          + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")"
          + "Frame rate: " + str(cap.get(cv2.CAP_PROP_FPS)))
    return cap


def detect(gray, frame):
    body = haarcascade_head_and_shoulders.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, flags=0,
                                                  minSize=(20, 20), maxSize=(150, 150))

    (rects, weights) = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.01, useMeanshiftGrouping=False)

    print("Bodies Up:", len(body))
    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

    print("People:", len(rects))
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)



    return frame


def show_video(cap, original, grayscal):
    while True:
        _, frame = cap.read()
        # frame = cv2.GaussianBlur(frame, (3, 3), 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # set when opencv is builded with Cuda 8.0
        # gray = cv2.fastNlMeansDenoisingMulti(gray, 2, 5, None, 4, 7, 35)


        if original:
            cv2.imshow('Original', frame)

        if grayscal:
            cv2.imshow('Video', gray)

        frame = detect(gray, frame)
        cv2.imshow("Output", frame)


        if cv2.waitKey(1) == 27:
            # print("Found {0} faces!".format(len(faces)))
            print("[INFO] cleaning up...")
            cv2.destroyAllWindows()
            print('x')
            break
            # exit(0)

            if cv2.getWindowProperty('Output', 0) == -1:
                cv2.destroyAllWindows()
                print('x')
                break


if __name__ == '__main__':
    system_info(False)
    grayscal = False
    original = False
    cap = capture_video(True)
    show_video(cap, original, grayscal)
