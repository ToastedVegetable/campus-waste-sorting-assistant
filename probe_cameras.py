"""
Show each OpenCV camera index so you can find the external webcam.

Run:
    python3 probe_cameras.py

Press:
    n  next camera index
    p  previous camera index
    q  quit
"""

import cv2
import numpy as np


def open_camera(index):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def main():
    index = 0
    cap = open_camera(index)
    print("Press n=next, p=previous, q=quit")

    while True:
        ok = False
        frame = None
        if cap.isOpened():
            ok, frame = cap.read()

        if not ok or frame is None:
            frame = np.full((360, 640, 3), 255, dtype=np.uint8)
            cv2.putText(frame, f"Camera index {index}: not available",
                        (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"Camera index {index}",
                        (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "n: next   p: previous   q: quit",
                        (24, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Camera probe", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key in {ord("n"), ord("p")}:
            cap.release()
            if key == ord("n"):
                index += 1
            else:
                index = max(0, index - 1)
            print(f"Trying camera index {index}")
            cap = open_camera(index)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
