import numpy as np
import datetime
import cv2

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('http://camera.buffalotrace.com/mjpg/video.mjpg?timestamp=1507887365324')
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
rate = float(cap.get(cv2.CAP_PROP_FPS))

print("Frame default resolution: (" + str(width) + "; " + str(height) + ") \nFrame rate: " + str(rate))
rate = 10
print("Frame default resolution: (" + str(width) + "; " + str(height) + ") \nFrame rate: " + str(rate))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
time = datetime.datetime.now().strftime("%Y_%m_%d_T_%H_%M_%S")
file_name = './Videos/' + time + '_output.avi'
print("Start:", time)
out = cv2.VideoWriter(file_name, fourcc, rate, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 180)    # 180 to rotate up to down

        # write the flipped frame
        out.write(frame)
        print(cap.get(cv2.CAP_PROP_FPS))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
            print("Closing stream")
            break
        else:
            print("Recording")
    else:
        break

# Release everything if job is finished
print("End:", datetime.datetime.now().strftime("%Y_%m_%d_T_%H_%M_%S"))
cap.release()
out.release()
print("Video saved")
cv2.destroyAllWindows()
