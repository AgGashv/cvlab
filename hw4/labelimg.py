import cv2
import os

DATASET_PATH = r"C:\Users\admin\Desktop\MAI\CV\cvlab1\hw4\details_dataset"
SAVE_PATH = r"C:\Users\admin\Desktop\MAI\CV\cvlab1\hw4\labeled_dataset"

# Создаем структуру папок YOLO
os.makedirs(os.path.join(SAVE_PATH, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, 'labels', 'val'), exist_ok=True)


class BoundingBoxLabeler:
    def __init__(self, image_path):
        # Загружаем оригинальное изображение
        self.original_image = cv2.imread(image_path)
        self.original_h, self.original_w = self.original_image.shape[:2]

        # Масштабируем изображение для отображения на экране
        self.display_scale = 1.0
        self.display_image = self.resize_for_display(self.original_image)
        self.display_h, self.display_w = self.display_image.shape[:2]

        self.clone = self.display_image.copy()
        self.boxes = []  # Хранит боксы в координатах отображения
        self.current_box = []
        self.drawing = False
        self.class_id = 0  # для одного класса "деталь"

    def resize_for_display(self, image, max_width=1200, max_height=800):
        """Масштабирует изображение, чтобы оно помещалось на экране"""
        h, w = image.shape[:2]

        # Если изображение уже маленькое, не масштабируем
        if w <= max_width and h <= max_height:
            self.display_scale = 1.0
            return image

        # Вычисляем коэффициент масштабирования
        scale_x = max_width / w
        scale_y = max_height / h
        self.display_scale = min(scale_x, scale_y)

        new_w = int(w * self.display_scale)
        new_h = int(h * self.display_scale)

        resized = cv2.resize(image, (new_w, new_h))
        print(f"Изображение масштабировано: {w}x{h} -> {new_w}x{new_h} (коэф: {self.display_scale:.2f})")
        return resized

    def convert_to_original_coords(self, x, y):
        """Конвертирует координаты из окна отображения в оригинальные"""
        orig_x = int(x / self.display_scale)
        orig_y = int(y / self.display_scale)
        return orig_x, orig_y

    def draw_boxes(self):
        img = self.clone.copy()
        for box in self.boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_box[2] = x
            self.current_box[3] = y
            # Визуализируем текущий бокс
            display = self.draw_boxes()
            cv2.rectangle(display, (self.current_box[0], self.current_box[1]),
                          (self.current_box[2], self.current_box[3]), (0, 0, 255), 2)
            cv2.imshow('Labeler', display)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.boxes.append(self.current_box.copy())
            print(f"Добавлен бокс (на экране): {self.current_box}")

    def run(self):
        # Создаем полноэкранное окно с возможностью изменения размера
        cv2.namedWindow('Labeler', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Labeler', self.display_w, self.display_h)
        cv2.setMouseCallback('Labeler', self.mouse_callback)

        print("\n" + "=" * 50)
        print("ИНСТРУКЦИЯ:")
        print("=" * 50)
        print("Нажми и удерживай ЛКМ для рисования прямоугольника")
        print("Отпусти ЛКМ для завершения прямоугольника")
        print("Нажми 's' для сохранения и перехода к следующему")
        print("Нажми 'u' для отмены последнего бокса")
        print("Нажми 'n' для следующего изображения (без сохранения текущего)")
        print("Нажми 'q' для выхода")
        print("=" * 50 + "\n")

        while True:
            display = self.draw_boxes()
            cv2.imshow('Labeler', display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                if self.save_annotations():
                    break
            elif key == ord('u') and self.boxes:
                self.boxes.pop()
                print(f"Удален последний бокс. Осталось: {len(self.boxes)}")
            elif key == ord('n'):
                print("Пропускаем изображение (боксы не сохранены)")
                break
            elif key == ord('q'):
                print("Выход")
                return False
        return True

    def save_annotations(self):
        if not self.boxes:
            print("Нет боксов для сохранения!")
            return False

        # Конвертируем боксы из координат отображения в оригинальные
        h, w = self.original_h, self.original_w

        with open(self.annotation_path, 'w') as f:
            for box in self.boxes:
                x1_disp, y1_disp, x2_disp, y2_disp = box

                # Конвертируем в оригинальные координаты
                x1, y1 = self.convert_to_original_coords(x1_disp, y1_disp)
                x2, y2 = self.convert_to_original_coords(x2_disp, y2_disp)

                # Формат YOLO (центр, ширина, высота)
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                f.write(f"{self.class_id} {x_center} {y_center} {width} {height}\n")

        print(f"Сохранено {len(self.boxes)} боксов в {self.annotation_path}")
        return True


# Использование
images = [f for f in os.listdir(DATASET_PATH) if f.endswith('.jpg')]

print(f"Найдено изображений: {len(images)}")
print(f"Путь к датасету: {DATASET_PATH}")
print(f"Путь сохранения: {SAVE_PATH}")

for i, img_name in enumerate(images):
    print(f"\n{'=' * 50}")
    print(f"Изображение {i + 1}/{min(20, len(images))}: {img_name}")
    print(f"{'=' * 50}")

    # Исправлено: заменяем .jpg на .txt (не .png)
    annotation_filename = img_name.replace('.jpg', '.txt')
    labeler = BoundingBoxLabeler(os.path.join(DATASET_PATH, img_name))
    labeler.annotation_path = os.path.join(SAVE_PATH, 'labels', 'train', annotation_filename)

    if not labeler.run():
        print("Прервано пользователем")
        break

    # Копируем изображение в папку images/train
    import shutil

    src_img = os.path.join(DATASET_PATH, img_name)
    dst_img = os.path.join(SAVE_PATH, 'images', 'train', img_name)
    shutil.copy(src_img, dst_img)
    print(f"Изображение скопировано: {dst_img}")

    cv2.destroyAllWindows()

print("\nРазметка завершена!")
