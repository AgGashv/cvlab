import cv2
import numpy as np


def order_points(points):
    """
    Сортировка углов
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect


# --- Поиск экрана ---
def detect_tv_screen(image):
    """
    Поиск экрана
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    screen = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and area > max_area:
            screen = approx
            max_area = area

    if screen is None:
        return None

    return order_points(screen.reshape(4, 2).astype(np.float32))


def warp_image(base, overlay, dst_points):
    """
    Преобразование кадров видео в экран с правильной перспективой
    """
    h, w = overlay.shape[:2]

    src_points = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_points, dst_points)

    warped = cv2.warpPerspective(overlay, H, (base.shape[1], base.shape[0]))

    mask = np.zeros(base.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_points.astype(int), 255)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    result = np.where(mask == 255, warped, base)
    return result


def main():
    """
    Основная функция
    """
    video_path = input("Видео с телевизором: ")
    overlay_path = input("Видео для вставки: ")

    cap = cv2.VideoCapture(video_path)
    overlay_cap = cv2.VideoCapture(overlay_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        "result_video.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    last_points = None

    while True:
        ret, frame = cap.read()
        ret2, overlay_frame = overlay_cap.read()

        if not ret:
            break

        # если вставляемое видео закончилось — запускаем заново
        if not ret2:
            overlay_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, overlay_frame = overlay_cap.read()

        points = detect_tv_screen(frame)

        if points is None:
            points = last_points
        else:
            last_points = points

        if points is not None:
            frame = warp_image(frame, overlay_frame, points)

        cv2.imshow("Result", frame)
        writer.write(frame)

        key = cv2.waitKey(25) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    overlay_cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("Готово: result_video.mp4")


if __name__ == "__main__":
    main()
