import os
import shutil
import random

DATASET_PATH = r"C:\Users\admin\Desktop\MAI\CV\cvlab1\hw4\details_dataset"  # исходные изображения
LABELED_PATH = r"C:\Users\admin\Desktop\MAI\CV\cvlab1\hw4\labeled_dataset"  # размеченные данные
TRAIN_RATIO = 0.8  # 80% на обучение, 20% на валидацию

# Получаем все размеченные изображения
labeled_images = []
for f in os.listdir(os.path.join(LABELED_PATH, 'labels', 'train')):
    if f.endswith('.txt'):
        img_name = f.replace('.txt', '.jpg')
        labeled_images.append(img_name)

print(f"Найдено размеченных изображений: {len(labeled_images)}")

# Перемешиваем изображения
random.shuffle(labeled_images)
split_idx = int(len(labeled_images) * TRAIN_RATIO)

print(f"На обучение: {split_idx} изображений")
print(f"На валидацию: {len(labeled_images) - split_idx} изображений")

# Копируем файлы
for i, img_name in enumerate(labeled_images):
    # Путь к исходным файлам
    src_img = os.path.join(DATASET_PATH, img_name)
    src_label = os.path.join(LABELED_PATH, 'labels', 'train', img_name.replace('.jpg', '.txt'))

    # Проверяем, существует ли исходный файл
    if not os.path.exists(src_img):
        print(f"Предупреждение: файл {src_img} не найден!")
        continue

    if not os.path.exists(src_label):
        print(f"Предупреждение: файл {src_label} не найден!")
        continue

    # Определяем куда копировать
    if i < split_idx:
        # В обучающую выборку - копируем только изображение (метка уже в правильной папке)
        dst_img = os.path.join(LABELED_PATH, 'images', 'train', img_name)
        # Для обучения метку не копируем, она уже лежит в labels/train
        shutil.copy2(src_img, dst_img)
        print(f"Обучение: {img_name} скопировано")
    else:
        # В валидационную выборку - копируем и изображение, и метку
        dst_img = os.path.join(LABELED_PATH, 'images', 'val', img_name)
        dst_label = os.path.join(LABELED_PATH, 'labels', 'val', img_name.replace('.jpg', '.txt'))

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_label, dst_label)
        print(f"🔍 Валидация: {img_name} скопировано")

    if (i + 1) % 5 == 0:  # Каждые 5 файлов выводим прогресс
        print(f"Обработано {i + 1}/{len(labeled_images)} изображений")

print("\n Датасет подготовлен!")
print(f"Обучающие изображения: {len(os.listdir(os.path.join(LABELED_PATH, 'images', 'train')))}")
print(f"Валидационные изображения: {len(os.listdir(os.path.join(LABELED_PATH, 'images', 'val')))}")
print(f"Обучающие метки: {len(os.listdir(os.path.join(LABELED_PATH, 'labels', 'train')))}")
print(f"Валидационные метки: {len(os.listdir(os.path.join(LABELED_PATH, 'labels', 'val')))}")
