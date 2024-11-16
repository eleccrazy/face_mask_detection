"""
File: train_test_splitter.py

This module splits images from source directories into training, validation, and test sets for a multi-class 
classification task involving mask detection. It reads images from separate directories for each class 
('with_mask', 'without_mask', and 'mask_weared_incorrect') and splits them into the specified subsets.

Dependencies:
    - os: For directory and file path management.
    - shutil: For copying files to target directories.
    - sklearn.model_selection.train_test_split: For splitting the dataset into subsets.

Directory Structure:
    The module assumes the following initial structure:
    ./dataset/cropped_dataset/
    ├── with_mask/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── without_mask/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── mask_weared_incorrect/
        ├── image1.jpg
        ├── image2.jpg
        └── ...

    After running the module, the following structure will be created:
    ./dataset/processed_dataset/
    ├── MobileNet-SSD/
    │   ├── train/
    │   │   ├── with_mask/
    │   │   │   ├── train_image1.jpg
    │   │   │   └── ...
    │   │   ├── without_mask/
    │   │   │   ├── train_image1.jpg
    │   │   │   └── ...
    │   │   └── mask_weared_incorrect/
    │   │       ├── train_image1.jpg
    │   │       └── ...
    │   ├── validation/
    │   │   ├── with_mask/
    │   │   ├── without_mask/
    │   │   └── mask_weared_incorrect/
    │   └── test/
    │       ├── with_mask/
    │       ├── without_mask/
    │       └── mask_weared_incorrect/
    └── YOLO/
        ├── train/
        ├── validation/
        └── test/

Usage:
    - Run the module in a Python environment where the ./dataset/cropped_dataset folder structure exists.
    - The script will create the train/validation/test subfolders and split the images accordingly.

Example:
    To split images into training, validation, and test sets:
        python train_test_splitter.py

Outputs:
    - Images from each class directory ('with_mask', 'without_mask', and 'mask_weared_incorrect') 
      will be moved to the appropriate folders within:
      - ./dataset/processed_dataset/MobileNet-SSD/
      - ./dataset/processed_dataset/YOLO/.
"""
import os
import shutil
from sklearn.model_selection import train_test_split

def create_model_dirs(base_path: str, model_name: str) -> tuple[str, str, str]:
    """
    Creates directories for training, validation, and testing subsets for a given model.

    Args:
        base_path (str): The base directory where the model-specific directories will be created.
        model_name (str): The name of the model (e.g., "MobileNet-SSD", "YOLO").

    Returns:
        tuple[str, str, str]: Paths to the train, validation, and test directories, respectively.
    """
    train_dir = os.path.join(base_path, model_name, 'train')
    val_dir = os.path.join(base_path, model_name, 'validation')
    test_dir = os.path.join(base_path, model_name, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    return train_dir, val_dir, test_dir


def split_and_move_images(
    source_dir: str,
    train_dir: str,
    val_dir: str,
    test_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.1
) -> None: 
    """
    Splits images from a source directory into training, validation, and test subsets
    and moves them to their respective target directories.

    Args:
        source_dir (str): Path to the source directory containing class-specific folders with images.
        train_dir (str): Path to the target directory for training images.
        val_dir (str): Path to the target directory for validation images.
        test_dir (str): Path to the target directory for test images.
        test_size (float): Fraction of images to use for the test set (default is 0.2).
        val_size (float): Fraction of images to use for the validation set (default is 0.1).

    Returns:
        None
    """
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        train_images, temp_images = train_test_split(images, test_size=test_size + val_size, random_state=42)
        val_images, test_images = train_test_split(temp_images, test_size=test_size / (test_size + val_size), random_state=42)

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))
        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

if __name__ == "__main__":
    base_output_dir = './dataset/processed_dataset'
    cropped_dataset_dir = './dataset/cropped_dataset'

    mobilenet_dirs = create_model_dirs(base_output_dir, "MobileNet-SSD")
    yolo_dirs = create_model_dirs(base_output_dir, "YOLO")

    split_and_move_images(cropped_dataset_dir, *mobilenet_dirs)
    split_and_move_images(cropped_dataset_dir, *yolo_dirs)
