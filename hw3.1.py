import cv2
import numpy as np
import math
from typing import Optional, List, Tuple, Any


def sort_corners_clockwise(points: List[List[float]]) -> np.ndarray:
    """
    Упорядочивает четыре угловые точки в порядке:
    верхний-левый, верхний-правый, нижний-правый, нижний-левый

    Args:
        points: Список из 4 точек [[x1,y1], [x2,y2], ...]

    Returns:
        Упорядоченный массив точек shape (4,2)
    """
    points_array = np.array(points, dtype="float32")

    sum_coordinates = points_array.sum(axis=1)
    ordered_corners = np.zeros((4, 2), dtype="float32")

    ordered_corners[0] = points_array[np.argmin(sum_coordinates)]
    ordered_corners[2] = points_array[np.argmax(sum_coordinates)]

    diff_coordinates = np.diff(points_array, axis=1)
    ordered_corners[1] = points_array[np.argmin(diff_coordinates)]
    ordered_corners[3] = points_array[np.argmax(diff_coordinates)]

    return ordered_corners


def warp_to_frontal_view(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Исправляет перспективу QR-кода и возвращает выровненное изображение

    Args:
        image: Исходный кадр (изображение)
        corners: Упорядоченные 4 точки QR-кода

    Returns:
        Выровненное изображение размером 300x300
    """
    target_size = 300

    destination_points = np.array([
        [0, 0],
        [target_size - 1, 0],
        [target_size - 1, target_size - 1],
        [0, target_size - 1]
    ], dtype="float32")

    perspective_matrix = cv2.getPerspectiveTransform(corners, destination_points)
    warped_image = cv2.warpPerspective(image, perspective_matrix, (target_size, target_size))

    return warped_image


def calculate_rotation_angle(corners: np.ndarray) -> float:
    """
    Вычисляет угол поворота QR-кода на основе верхней границы

    Args:
        corners: Упорядоченные 4 точки [[top_left], [top_right], [bottom_right], [bottom_left]]

    Returns:
        Угол в градусах
    """
    (top_left, top_right, _, _) = corners
    delta_x = top_right[0] - top_left[0]
    delta_y = top_right[1] - top_left[1]
    return math.degrees(math.atan2(delta_y, delta_x))


def smooth_bounding_box(previous_box: Optional[np.ndarray], current_box: np.ndarray,
                        smoothing_factor: float = 0.7) -> np.ndarray:
    """
    Сглаживает координаты bounding box'а для уменьшения дрожания

    Args:
        previous_box: Предыдущий bounding box или None
        current_box: Текущий bounding box
        smoothing_factor: Коэффициент сглаживания (0-1, где 1 = максимальное сглаживание)

    Returns:
        Сглаженный bounding box
    """
    if previous_box is None:
        return current_box
    return smoothing_factor * previous_box + (1 - smoothing_factor) * current_box


def run_qr_scanner() -> None:
    """
    Главная функция: захват видео с камеры, обнаружение и декодирование QR-кодов,
    исправление перспективы и отображение результатов
    """
    video_capture = cv2.VideoCapture(0)
    qr_detector = cv2.QRCodeDetector()

    previous_bounding_box = None

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        decoded_data, bounding_box, _ = qr_detector.detectAndDecode(frame)

        # Отображение режима работы
        cv2.putText(frame, "Mode: CORRECTED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if bounding_box is not None:
            bounding_box = bounding_box.astype(np.float32)

            # Сглаживание bounding box'а
            bounding_box = smooth_bounding_box(previous_bounding_box, bounding_box)
            previous_bounding_box = bounding_box

            sorted_corners = sort_corners_clockwise(bounding_box[0])

            # Рисуем рамку вокруг QR-кода
            for i in range(4):
                start_point = tuple(sorted_corners[i].astype(int))
                end_point = tuple(sorted_corners[(i + 1) % 4].astype(int))
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)

            # Коррекция перспективы для отображения
            corrected_view = warp_to_frontal_view(frame, sorted_corners)

            rotation_angle = calculate_rotation_angle(sorted_corners)

            # Подготовка текста с данными QR-кода
            data_text = f"DATA: {decoded_data}" if decoded_data else "DATA: ---"
            cv2.putText(frame, data_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Отображение угла поворота
            angle_text = f"ANGLE: {rotation_angle:.2f}"
            cv2.putText(frame, angle_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Показываем исправленное изображение
            cv2.imshow("Corrected QR", corrected_view)

        # Показываем основной кадр с разметкой
        cv2.imshow("QR Scanner", frame)

        # Выход по клавише ESC
        if cv2.waitKey(1) == 27:
            break

    # Освобождаем ресурсы
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_qr_scanner()
