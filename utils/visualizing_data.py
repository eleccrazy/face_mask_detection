"""
Module: visualizing_data.py

This module visualizes the data distribution across train, validation, and test datasets 
for different models (e.g., MobileNet-SSD and YOLO) used in a face mask detection project.

Functionality:
    - Counts the number of images in each class directory (e.g., 'with_mask', 'without_mask', 'mask_weared_incorrect') 
      across train, validation, and test splits.
    - Plots the data distribution for each model, showing the number of images in each class 
      for the respective splits.

Dependencies:
    - os: For file system operations.
    - matplotlib: For plotting the data distribution.

Expected Directory Structure:
    ./dataset/processed_dataset/
    ├── MobileNet-SSD/
    │   ├── train/
    │   │   ├── with_mask/
    │   │   ├── without_mask/
    │   │   └── mask_weared_incorrect/
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
        │   ├── with_mask/
        │   ├── without_mask/
        │   └── mask_weared_incorrect/
        ├── validation/
        │   ├── with_mask/
        │   ├── without_mask/
        │   └── mask_weared_incorrect/
        └── test/
            ├── with_mask/
            ├── without_mask/
            └── mask_weared_incorrect/

Functions:
    - count_images_in_directory(directory):
        Counts the number of images in each class directory for a specific dataset split.

    - get_data_distribution(base_dir):
        Aggregates the counts of images across train, validation, and test splits for a given model.

    - plot_data_distribution(data_counts, model_name):
        Plots the distribution of images for a specific model, showing counts for each class 
        ('with_mask', 'without_mask', 'mask_weared_incorrect') across train, validation, and test splits.

Usage:
    - Ensure the processed dataset directories exist as per the above structure.
    - Run this module to count the images and visualize the data distribution for 
      MobileNet-SSD and YOLO models.

Example:
    To visualize data distribution:
        python visualizing_data.py

Outputs:
    - Console: Prints the data distribution as dictionaries for each model and split.
    - Plots: Displays bar charts showing the data distribution for each split 
      (train, validation, test) and class ('with_mask', 'without_mask', 'mask_weared_incorrect').

Expected output for the Face Mask Detection Dataset:
    MobileNet-SSD Data Distribution: {'train': {'with_mask': 2262, 'without_mask': 501, 'mask_weared_incorrect': 86}, 
                                       'validation': {'with_mask': 323, 'without_mask': 72, 'mask_weared_incorrect': 12}, 
                                       'test': {'with_mask': 647, 'without_mask': 144, 'mask_weared_incorrect': 25}}
    YOLO Data Distribution: {'train': {'with_mask': 2262, 'without_mask': 501, 'mask_weared_incorrect': 86}, 
                              'validation': {'with_mask': 323, 'without_mask': 72, 'mask_weared_incorrect': 12}, 
                              'test': {'with_mask': 647, 'without_mask': 144, 'mask_weared_incorrect': 25}}

Note:
    - If directories are missing or empty, the module will print warnings and plot 
      "No Data" for the corresponding splits.
"""
import os
import matplotlib.pyplot as plt
from typing import Dict


def count_images_in_directory(directory: str) -> Dict[str, int]:
    """
    Counts the number of images in each class subdirectory within a given directory.

    Args:
        directory (str): The path to the parent directory containing class subdirectories.

    Returns:
        Dict[str, int]: A dictionary where the keys are class names (subdirectory names)
                        and the values are the counts of images in each class.
    """
    class_counts = {}
    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        if os.path.isdir(class_path):
            num_images = len([file for file in os.listdir(class_path) if file.endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_dir] = num_images
    return class_counts

def get_data_distribution(base_dir: str)-> Dict[str, Dict[str, int]]:
    """
    Aggregates the distribution of images across train, validation, and test splits.

    Args:
        base_dir (str): The base directory containing train, validation, and test subdirectories.

    Returns:
        Dict[str, Dict[str, int]]: A dictionary where the keys are split names ('train', 'validation', 'test')
                                   and the values are dictionaries containing class-wise image counts for each split.
    """
    splits = ['train', 'validation', 'test']
    data_distribution = {}

    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            data_distribution[split] = {}
            continue
        data_distribution[split] = count_images_in_directory(split_dir)

    return data_distribution

def plot_data_distribution(data_counts: Dict[str, Dict[str, int]], model_name: str) -> None:
    """
    Plots the distribution of images for a specific model, showing counts for each class
    ('with_mask', 'without_mask', 'mask_weared_incorrect') across train, validation, and test splits.

    Args:
        data_counts (Dict[str, Dict[str, int]]): A dictionary containing class-wise image counts
                                                 for train, validation, and test splits.
        model_name (str): The name of the model (e.g., "MobileNet-SSD", "YOLO").

    Returns:
        None
    """
    splits = ['train', 'validation', 'test']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, split in enumerate(splits):
        split_counts = data_counts.get(split, {})
        if not split_counts:
            axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
            axes[i].set_title(f'{model_name} - {split.capitalize()} Data Distribution')
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel('Number of Images')
            continue

        classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']
        counts = [split_counts.get(cls, 0) for cls in classes]
        colors = ['skyblue', 'lightgreen', 'lightcoral']

        axes[i].bar(classes, counts, color=colors)
        axes[i].set_title(f'{model_name} - {split.capitalize()} Data Distribution')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Number of Images')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    mobilenet_base_dir = './dataset/processed_dataset/MobileNet-SSD'
    yolo_base_dir = './dataset/processed_dataset/YOLO'

    mobilenet_counts = get_data_distribution(mobilenet_base_dir)
    yolo_counts = get_data_distribution(yolo_base_dir)

    print("MobileNet-SSD Data Distribution:", mobilenet_counts)
    print("YOLO Data Distribution:", yolo_counts)

    plot_data_distribution(mobilenet_counts, "MobileNet-SSD")
    plot_data_distribution(yolo_counts, "YOLO")
