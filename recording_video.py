import numpy as np
import cv2

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('http://camera.buffalotrace.com/mjpg/video.mjpg?timestamp=1507887365324')
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
rate = float(cap.get(cv2.CAP_PROP_FPS))

print("Frame default resolution: (" + str(width) + "; " + str(height) + ") \nFrame rate: " + str(rate))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./Videos/output.avi', fourcc, rate, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 180)    # 180 to rotate up to down

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27:
            print("Closing stream")
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
print("Video saved")
cv2.destroyAllWindows()
