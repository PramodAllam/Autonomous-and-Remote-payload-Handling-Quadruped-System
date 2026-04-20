import cv2

cap = cv2.VideoCapture("http://192.168.137.195/stream")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[Warning] Frame grab failed, retrying...")
        continue  # Skip this loop

    print(frame.shape)
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
