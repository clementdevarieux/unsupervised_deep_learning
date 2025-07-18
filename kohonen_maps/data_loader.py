import os
import cv2
import numpy as np

def load_images_from_folder(folder, target_size=(32, 32), max_images_per_class=200):
    images = []
    labels = []
    class_names = os.listdir(folder)

    for class_name in class_names:
        class_folder = os.path.join(folder, class_name)
        if os.path.isdir(class_folder):
            image_files = [f for f in os.listdir(class_folder) 
                          if f.lower().endswith('.png')]
            
            selected_files = image_files[:max_images_per_class]

            for filename in selected_files:
                img_path = os.path.join(class_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, target_size)  
                    images.append(img)
                    labels.append(class_name)

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