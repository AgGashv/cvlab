from ultralytics import YOLO
import os

DATA_YAML = r"C:\Users\admin\Desktop\MAI\CV\cvlab1\hw4\labeled_dataset\dataset.yaml"
PROJECT_DIR = r"C:\Users\admin\Desktop\MAI\CV\cvlab1\hw4\runs\detect"


def train_custom_model():
    # Создаем папку, если её нет
    os.makedirs(PROJECT_DIR, exist_ok=True)

    # Используем предобученную модель
    model = YOLO("yolov8n.pt")

    # Обучение
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=16,
        project=PROJECT_DIR,
        name="details_detector",
        patience=20,
        augment=True,
        workers=4,
        device='cpu'
    )

    best_weights = os.path.join(PROJECT_DIR, "details_detector", "weights", "best.pt")
    print(f"Модель сохранена: {best_weights}")
    return best_weights


if __name__ == "__main__":
    train_custom_model()
