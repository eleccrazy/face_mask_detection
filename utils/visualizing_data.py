"""
Module: visualizing_data.py

This module visualizes the data distribution across train, validation, and test datasets 
for different models (e.g., MobileNet-SSD and YOLO) used in a face mask detection project.

Functionality:
    - Counts the number of images in each class directory (e.g., 'mask', 'no_mask') 
      across train, validation, and test splits.
    - Plots the data distribution for each model, showing the number of images in each 
      class for the respective splits.

Dependencies:
    - os: For file system operations.
    - matplotlib: For plotting the data distribution.

Expected Directory Structure:
    ./dataset/processed_dataset/
    ├── MobileNet-SSD/
    │   ├── train/
    │   │   ├── mask/
    │   │   └── no_mask/
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

Functions:
    - count_images_in_split(base_dir):
        Counts the number of images in each class for train, validation, and test splits.

    - plot_data_distribution(data_counts, model_name):
        Plots the distribution of images for a specific model, showing counts for each 
        class (e.g., 'mask', 'no_mask') across train, validation, and test splits.

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
      (train, validation, test) and class ('mask', 'no_mask').

Expected output for the Face Mask Detection Dataset:
    MobileNet-SSD Data Distribution: {'train': {'mask': 537, 'no_mask': 59}, 'validation': {'mask': 77, 'no_mask': 8}, 'test': {'mask': 154, 'no_mask': 18}}
    YOLO Data Distribution: {'train': {'mask': 537, 'no_mask': 59}, 'validation': {'mask': 77, 'no_mask': 8}, 'test': {'mask': 154, 'no_mask': 18}}

Note:
    - If directories are missing or empty, the module will print warnings and plot 
      "No Data" for the corresponding splits.
"""
import os
import matplotlib.pyplot as plt

# Paths to processed dataset directories
mobilenet_base_dir = './dataset/processed_dataset/MobileNet-SSD'
yolo_base_dir = './dataset/processed_dataset/YOLO'


# Function to count images in each class directory for a given split
def count_images_in_split(base_dir):
    """
    Counts images in the train, validation, and test directories for each class.

    Parameters:
        base_dir (str): Base directory containing train, validation, and test splits.

    Returns:
        dict: A dictionary with counts of images for each split and class.
    """
    splits = ['train', 'validation', 'test']
    data_distribution = {}

    for split in splits:
        split_dir = os.path.join(base_dir, split)
        split_counts = {}
        if not os.path.exists(split_dir):  # Check if the split directory exists
            print(f"Warning: {split_dir} does not exist.")
            continue

        for class_dir in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_dir)
            if os.path.isdir(class_path):
                num_images = len([
                    file for file in os.listdir(class_path)
                    if file.endswith(('.jpg', '.jpeg', '.png'))
                ])
                split_counts[class_dir] = num_images

        data_distribution[split] = split_counts

    return data_distribution


# Count images for MobileNet-SSD and YOLO
mobilenet_counts = count_images_in_split(mobilenet_base_dir)
yolo_counts = count_images_in_split(yolo_base_dir)


# Function to plot the distribution
def plot_data_distribution(data_counts, model_name):
    """
    Plots data distribution for a given model.

    Parameters:
        data_counts (dict): Dictionary with counts of images for each split and class.
        model_name (str): Name of the model (e.g., "MobileNet-SSD", "YOLO").
    """
    splits = ['train', 'validation', 'test']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, split in enumerate(splits):
        split_counts = data_counts.get(split, {})
        if not split_counts:  # Check if the split contains any data
            axes[i].text(0.5, 0.5, 'No Data', ha='center',
                         va='center', fontsize=12)
            axes[i].set_title(
                f'{model_name} - {split.capitalize()} Data Distribution')
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel('Number of Images')
            continue

        colors = ['skyblue', 'lightgreen',
                  'lightcoral', 'orange'][:len(split_counts)]
        axes[i].bar(split_counts.keys(), split_counts.values(), color=colors)
        axes[i].set_title(
            f'{model_name} - {split.capitalize()} Data Distribution')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Number of Images')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Plot data distribution for both models
    print("MobileNet-SSD Data Distribution:", mobilenet_counts)
    print("YOLO Data Distribution:", yolo_counts)

    plot_data_distribution(mobilenet_counts, "MobileNet-SSD")
    plot_data_distribution(yolo_counts, "YOLO")
