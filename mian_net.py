# Title: Detection and classification of moving objects based on the real-time camera image.
# ©WinczewskiDamian2017

import numpy as np
import time
import cv2
import sys
import torch
from torch.autograd import Variable
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import torchvision
# vgg19 = torchvision.models.vgg19(pretrained=True)
# torch.save(vgg19, 'vgg19.pth')
# net.load_state_dict(torch.load('vgg19.pth'))


# Creating the SSD neural network
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location=lambda storage, loc: storage))
# Creating the transformation
transform = BaseTransform(net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))


def system_info(cv2_info):
    print("The Python version is %s.%s.%s" % sys.version_info[:3])
    print("The OpenCV version is", cv2.__version__)
    print("Cuda in Torch is available: ", torch.cuda.is_available())
    if cv2_info:
        print("Cv2 build information", cv2.getBuildInformation())
    # print("Models trained, are detecting:", labelmap[:])

def capture_video(my_camera):
    if my_camera:
        # cap = cv2.VideoCapture(0) #camera systemowa
        cap = cv2.VideoCapture('./Videos/2017_11_27_T_12_47_28_output.avi')
        # cap = cv2.VideoCapture('./Videos/dev_stream.avi')
    else:
        cap = cv2.VideoCapture('http://camera.buffalotrace.com/mjpg/video.mjpg?timestamp=1507887365324')

    print("Frame default resolution: ("
          + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + "; "
          + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + ")"
          + "Frame rate: " + str(cap.get(cv2.CAP_PROP_FPS)))

    return cap


def detect(frame, net, transform, time):  # add tracking
    people = 0
    objects = []
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    # detections = [batch, number of classes, number of occurence, (score, x0, Y0, x1, y1)]
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.5 and labelmap[i - 1] == 'person':
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)

            cv2.circle(frame, (int(pt[0] + (pt[2] - pt[0]) / 2), int(pt[1] + (pt[3] - pt[1]) / 2)), 5, (0, 255, 0), -1)
            # cv2.circle(tracking, (int(pt[0] + (pt[2] - pt[0])/2), int(pt[1] + (pt[3] - pt[1])/2)), 2, (0, 255, 0), -1)

            cv2.putText(frame, str(labelmap[i - 1]) + " " + str(people), (int(pt[0]), int(pt[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            # print(str(labelmap[i - 1]) + " " + str(i) + ": " + str(detections[0, i, j, 0]*100))
            j += 1

            objects.append([people, time, int(pt[0] + (pt[2] - pt[0]) / 2), int(pt[1] + (pt[3] - pt[1]) / 2)])
            people += 1

    # print(str(people + 1) + " people detected.")
    return frame, objects  # tracking


def ploting(persons):
    print("P: ", persons)
    print("Detected: ", max(persons[:, 0]) + 1, " people")

    plots = []
    for i in range(max(persons[:, 0]) + 1):
        one_object = []
        for w in persons:
            if w[0] == i:
                one_object.append(w[1:])
        plots.append(one_object)

    plots = np.asarray(plots)
    colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])

    print("Prepering plot")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    ax.set_ylim(0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ax.set_zlim(0, max(persons[:, 1]))
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Time axis')

    i = 0
    for person in plots:
        person = np.array(person)
        ax.plot(person[:, 1], person[:, 2], person[:, 0], str(colors[i] + '-'), label=str('person ' + str(i)))
        ax.plot(person[:, 1], person[:, 2], person[:, 0], str(colors[i] + 'o'))
        if i > len(colors) - 1:
            i = 0
        else:
            i += 1
    ax.legend()
    plt.show()
    return 0


def show_video(cap):
    # tracking = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), np.uint8)
    time = 0
    data = []
    persons = []
    print("P:", persons)
    print("Data", data)
    while True:
        _, frame = cap.read()
        frame, objects = detect(frame, net.eval(), transform, time)
        # frame, tracking = detect(frame, net.eval(), transform, tracking)
        cv2.imshow("Output", frame)
        # cv2.imshow("Tracking", tracking)
        time += 1

        if objects:
            for people in objects:
                print("Person", people[0], "T:", people[1], "X:", people[2], "Y:", people[3])
                data.append(people)

        if cv2.waitKey(1) == 27:
            print("[INFO] cleaning up...")
            persons = np.array(data)
            ploting(persons)
            cv2.destroyAllWindows()
            print('x')
            break
            if cv2.getWindowProperty('Output', 0) == -1:
                cv2.destroyAllWindows()
                print('x')
                break


if __name__ == '__main__':
    system_info(False)
    cap = capture_video(False)
    show_video(cap)
