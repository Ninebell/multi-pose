import cv2

if __name__ == "__main__":
    t = cv2.VideoCapture(0)
    while True:
        if cv2.waitKey(10) == 27:
            break
        img = t.grab()
        cv2.imshow("test",img)