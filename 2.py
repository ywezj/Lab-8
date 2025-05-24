import time
import numpy as np
import cv2

def video_processing():
    cap = cv2.VideoCapture(0)
    down_points = (640, 480)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)
        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            obj_center = (x + w // 2, y + h // 2)
            cv2.circle(frame, obj_center, 5, (0, 0, 255), -1)

            distance = np.sqrt((frame_center[0] - obj_center[0]) ** 2 + (frame_center[1] - obj_center[1]) ** 2)

            cv2.putText(frame, f"Distance: {distance:.1f} px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if i % 5 == 0:
                print("Object center:", obj_center, "| Distance:", distance)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)
        i += 1

    cap.release()


if __name__== 'main':
    video_processing()

cv2.waitKey(0)
cv2.destroyAllWindows()

