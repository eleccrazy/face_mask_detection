"""
This module splits images from source directories into training, validation, and test sets for a binary
classification task involving mask detection. It reads images from separate directories for each 
class ('mask' and 'no_mask') and splits them into training, validation, and test subsets.

Dependencies:
    - os: For directory and file path management.
    - shutil: For copying files to target directories.
    - sklearn.model_selection.train_test_split: For splitting the dataset into subsets.

Directory Structure:
    The module assumes the following initial structure:
    ./dataset/binary_dataset/
    ├── mask/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── no_mask/
        ├── image1.jpg
        ├── image2.jpg
        └── ...

    After running the module, the following structure will be created:
    ./dataset/processed_dataset/
    ├── MobileNet-SSD/
    │   ├── train/
    │   │   ├── mask/
    │   │   │   ├── train_image1.jpg
    │   │   │   └── ...
    │   │   └── no_mask/
    │   │       ├── train_image1.jpg
    │   │       └── ...
    │   ├── validation/
    │   │   ├── mask/
    │   │   └── no_mask/
    │   └── test/
    │       ├── mask/
    │       └── no_mask/
    └── YOLO/
        ├── train/
        ├── validation/
        └── test/

Usage:
    - Run the module in a Python environment where the ./dataset folder structure exists.
    - The script will create the train/validation/test subfolders and split the images accordingly.

Example:
    To split images into training, validation, and test sets:
        python train_test_splitter.py

Outputs:
    - Images from each class directory ('mask' and 'no_mask') will be moved to the appropriate 
      folders within ./dataset/processed_dataset/MobileNet-SSD/ and ./dataset/processed_dataset/YOLO/.
"""
import os
import shutil
import cv2
from sklearn.model_selection import train_test_split
from typing import Tuple, List

# Define source directories
source_mask_dir = './dataset/binary_dataset/mask'
source_no_mask_dir = './dataset/binary_dataset/no_mask'

# Define base output directory
base_output_dir = './dataset/processed_dataset'


def create_model_dirs(base_path: str, model_name: str) -> Tuple[str, str, str]:
    """
    Creates directories for train, validation, and test subsets for a given model.

    Parameters:
        base_path (str): The base path where the model's directories will be created.
        model_name (str): The name of the model (e.g., "MobileNet-SSD", "YOLO").

    Returns:
        tuple: Paths to train, validation, and test directories.
    """
    train_dir = os.path.join(base_path, model_name, "train")
    val_dir = os.path.join(base_path, model_name, "validation")
    test_dir = os.path.join(base_path, model_name, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    return train_dir, val_dir, test_dir


def split_and_move_images(source_dir: str, train_dir: str, val_dir: str, test_dir: str,
                          class_name: str, test_size: float = 0.2, val_size: float = 0.1) -> None:
    """
    Splits images into train, validation, and test sets and moves them to respective directories.

    Parameters:
        source_dir (str): Directory containing source images to be split.
        train_dir (str): Directory where training images will be stored.
        val_dir (str): Directory where validation images will be stored.
        test_dir (str): Directory where test images will be stored.
        class_name (str): Name of the class (e.g., 'mask', 'no_mask').
        test_size (float): Fraction of images to use for the test set.
        val_size (float): Fraction of images to use for the validation set.

    Returns:
        None
    """
    images = os.listdir(source_dir)

    # First split: train and temp (validation + test)
    train_images, temp_images = train_test_split(
        images, test_size=test_size + val_size, random_state=42)

    # Second split: validation and test
    val_images, test_images = train_test_split(
        temp_images, test_size=test_size / (test_size + val_size), random_state=42)

    # Ensure class directories exist in train, validation, and test
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Move training images
    for img in train_images:
        shutil.copy(os.path.join(source_dir, img),
                    os.path.join(train_dir, class_name, img))

    # Move validation images
    for img in val_images:
        shutil.copy(os.path.join(source_dir, img),
                    os.path.join(val_dir, class_name, img))

    # Move testing images
    for img in test_images:
        shutil.copy(os.path.join(source_dir, img),
                    os.path.join(test_dir, class_name, img))


def resize_and_save_images(image_dir: str, target_dir: str,
                           target_size: Tuple[int, int]) -> None:
    """
    Resizes images in a directory to the target size and saves them to the specified target directory.

    Parameters:
        image_dir (str): Directory containing the images to be resized.
        target_dir (str): Directory where resized images will be saved.
        target_size (tuple): Target size (width, height) for resizing.
    """
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                resized_img = cv2.resize(img, target_size)

                # Save resized image to the corresponding directory
                relative_path = os.path.relpath(root, image_dir)
                save_dir = os.path.join(target_dir, relative_path)
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, file), resized_img)


def prepare_data() -> None:
    """
    Prepares datasets for MobileNet-SSD and YOLO by:
    - Splitting images into train, validation, and test sets.
    - Resizing images for each model and saving them in respective directories.

    Returns:
        None
    """
    # Create directories for both models
    mobilenet_train, mobilenet_val, mobilenet_test = create_model_dirs(
        base_output_dir, "MobileNet-SSD")
    yolo_train, yolo_val, yolo_test = create_model_dirs(
        base_output_dir, "YOLO")

    # Split and move images for both classes
    split_and_move_images(source_mask_dir, mobilenet_train, mobilenet_val,
                          mobilenet_test, class_name="mask")
    split_and_move_images(source_no_mask_dir, mobilenet_train, mobilenet_val,
                          mobilenet_test, class_name="no_mask")

    # Resize images for MobileNet-SSD
    resize_and_save_images(mobilenet_train, mobilenet_train, (224, 224))
    resize_and_save_images(mobilenet_val, mobilenet_val, (224, 224))
    resize_and_save_images(mobilenet_test, mobilenet_test, (224, 224))

    # Resize images for YOLO
    resize_and_save_images(mobilenet_train, yolo_train, (416, 416))
    resize_and_save_images(mobilenet_val, yolo_val, (416, 416))
    resize_and_save_images(mobilenet_test, yolo_test, (416, 416))

    print("Dataset prepared for MobileNet-SSD and YOLO.")


if __name__ == "__main__":
    # Run the preparation process
    prepare_data()
