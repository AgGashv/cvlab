import sys
import cv2
from tkinter import Tk, filedialog
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\admin\Desktop\MAI\CV\cvlab1\hw4\runs\detect\details_detector\weights\best.pt"


def run_camera_detection_optimized(model, conf=0.6, camera_index=0):
    """
    Оптимизированная версия с фильтрацией ложных срабатываний
    """
    cap = cv2.VideoCapture(camera_index)

    # Настройки камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Не удалось открыть камеру.")
        return

    print("Камера запущена")
    # print("Нажми 'Q' для выхода, '+'/'-' для изменения порога")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Предсказание
        results = model(frame, conf=conf, verbose=False)

        # Создаем копию кадра для рисования
        display_frame = frame.copy()

        if results[0].boxes:
            for box in results[0].boxes:
                # Получаем координаты
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Фильтр по размеру (игнорируем слишком маленькие объекты)
                box_width = x2 - x1
                box_height = y2 - y1
                area_ratio = (box_width * box_height) / (frame.shape[0] * frame.shape[1])

                # Игнорируем объекты меньше 1% или больше 80% кадра
                if area_ratio < 0.01 or area_ratio > 0.8:
                    continue

                # Игнорируем объекты у краев кадра (обычно артефакты)
                margin = 20
                if x1 < margin or y1 < margin or x2 > frame.shape[1] - margin or y2 > frame.shape[0] - margin:
                    continue

                # Рисуем только отфильтрованные детекции
                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)  # Зеленый для уверенных, желтый для средних
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                # Добавляем текст с уверенностью
                label = f'detail {confidence:.2f}'
                cv2.putText(display_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("YOLO Detection - Детали", display_frame)

        # Управление
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def choose_image_file():
    """Открывает стандартное окно выбора файла Windows."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
    )

    root.destroy()
    return file_path if file_path else None


def run_image_detection(model, conf=0.6):  # Повышен порог для изображений
    """
    Выбор изображения через проводник Windows и распознавание на нём.
    """
    image_path = choose_image_file()

    if not image_path:
        print("Файл не выбран.")
        return

    print(f"Выбрано изображение: {image_path}")

    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print("Не удалось открыть изображение.")
        return

    # Предсказание
    results = model.predict(image, conf=conf, verbose=False)

    # Картинка с рамками
    annotated_image = results[0].plot()

    # Изменяем размер для отображения, если нужно
    h, w = annotated_image.shape[:2]
    if w > 1200 or h > 800:
        scale = min(1200 / w, 800 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        annotated_image = cv2.resize(annotated_image, (new_w, new_h))

    cv2.imshow("YOLO Image Detection", annotated_image)
    print("Нажми любую клавишу в окне изображения, чтобы закрыть его.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    model = YOLO(MODEL_PATH)

    while True:
        print("\n" + "=" * 40)
        print("ВЫБЕРИТЕ РЕЖИМ:")
        print("=" * 40)
        print("1 - Распознавание с камеры")
        print("2 - Выбрать изображение и распознать")
        print("0 - Выход")
        print("=" * 40)

        choice = input("Введите номер: ").strip()

        if choice == "1":
            run_camera_detection_optimized(model, conf=0.85, camera_index=0)
        elif choice == "2":
            run_image_detection(model, conf=0.6)
        elif choice == "0":
            print("Выход.")
            sys.exit()
        else:
            print("Неверный выбор. Попробуй ещё раз.")


if __name__ == "__main__":
    main()
