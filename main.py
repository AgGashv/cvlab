import cv2
import sys
import tkinter as tk
from threading import Thread

rectangles = []

RECT_SIZE = 20


def mouse_callback(event, x, y, flags, param):
    """
    Создание прямоугольников при нажатии левой кнопки мыши
    """
    global rectangles

    if event == cv2.EVENT_LBUTTONDOWN:
        rectangles.append((x - RECT_SIZE, y - RECT_SIZE,
                           x + RECT_SIZE, y + RECT_SIZE))


def video_loop(source):
    """
    Отображение веб-камеры и взаимодействие с этим окном
    """
    global rectangles

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Ошибка открытия видео")
        return

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", mouse_callback)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        for (x1, y1, x2, y2) in rectangles:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Video", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            rectangles.clear()
        elif cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) <= 0:
            break

    cap.release()
    cv2.destroyAllWindows()


def start_video():
    """
    Начало отображения веб-камеры
    """
    source = 0

    video_loop(source)


def create_gui():
    """
    Создание интерфейса на tkinter
    """
    root = tk.Tk()
    root.title("Control Panel")

    btn_start = tk.Button(root, text="Start Video", command=lambda: Thread(target=start_video).start())
    btn_start.pack(pady=10)

    btn_quit = tk.Button(root, text="Quit", command=root.quit)
    btn_quit.pack(pady=10)

    root.bind('q', lambda event: root.quit())
    root.focus_set()

    root.mainloop()


if __name__ == "__main__":
    create_gui()
