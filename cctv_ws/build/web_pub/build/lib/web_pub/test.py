import cv2

cap = cv2.VideoCapture(2)  # /dev/video2

ret, frame = cap.read()
if not ret:
    print("fail to read")
else:
    print("frame shape:", frame.shape)  # (720, 1280, 3) 기대

cap.release()
