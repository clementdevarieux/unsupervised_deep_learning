import os
import cv2
import numpy as np
import random


def load_images_from_folder(folder, target_size=(32, 32), max_images_per_class=200, n_classes=5):
    """
    Load images from folder structure, handling both nested and direct image folders.
    Randomly samples n_classes from available classes.
    """
    images = []
    labels = []

    # First, collect all available classes
    all_classes = []

    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            # Check if this folder has subfolders (like Apple/Apple A) or direct images
            subitems = os.listdir(item_path)
            has_subfolders = any(os.path.isdir(os.path.join(item_path, subitem)) for subitem in subitems)

            if has_subfolders:
                # Handle nested structure (like Apple/Apple A, Apple/Apple B)
                for subfolder in subitems:
                    subfolder_path = os.path.join(item_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        # Check if this subfolder contains images
                        image_files = [f for f in os.listdir(subfolder_path)
                                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        if image_files:
                            all_classes.append((subfolder_path, f"{item}_{subfolder}"))
            else:
                # Handle direct image structure (like Carambola with images directly)
                image_files = [f for f in os.listdir(item_path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    all_classes.append((item_path, item))

    # Randomly sample n_classes
    if len(all_classes) > n_classes:
        selected_classes = random.sample(all_classes, n_classes)
    else:
        selected_classes = all_classes
        print(f"Warning: Only {len(all_classes)} classes available, using all of them.")

    print(f"Selected classes: {[class_name for _, class_name in selected_classes]}")

    # Load images from selected classes
    for class_folder, class_name in selected_classes:
        image_files = [f for f in os.listdir(class_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Randomly sample images if there are more than max_images_per_class
        if len(image_files) > max_images_per_class:
            selected_files = random.sample(image_files, max_images_per_class)
        else:
            selected_files = image_files

        print(f"Loading {len(selected_files)} images from {class_name}")

        for filename in selected_files:
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                images.append(img)
                labels.append(class_name)

    print(f"Total images loaded: {len(images)}")
    return images, labels


def normalize_images(images):
    return [img / 255.0 for img in images]


def load_and_normalize_images(images, flatten=True):
    normalized_images = []
    for img in images:
        normalized_image = img.astype(np.float32) / 255.0
        if flatten:
            normalized_image = normalized_image.flatten()
        normalized_images.append(normalized_image)

    return normalized_images