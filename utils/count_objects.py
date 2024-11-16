"""
Module: count_objects.py

This module parses PASCAL VOC-style XML annotation files to count the total number of labeled objects 
(people) in the dataset and their distribution across different classes (e.g., 'with_mask', 'without_mask', 
'mask_weared_incorrect').

Functionality:
    - Iterates through all XML annotation files in a specified directory.
    - Extracts object class names from each annotation file.
    - Aggregates the count of objects for each class and calculates the total number of objects.

Dependencies:
    - os: For directory and file path management.
    - xml.etree.ElementTree: For parsing XML annotation files.

Example Use Case:
    This module is useful for analyzing the dataset before training, providing insight into the class distribution
    and ensuring there is sufficient data for each class.

Expected Input:
    A folder containing XML annotation files in PASCAL VOC format with the following structure:
    <annotation>
        <object>
            <name>with_mask</name>
            <bndbox>
                <xmin>79</xmin>
                <ymin>105</ymin>
                <xmax>109</xmax>
                <ymax>142</ymax>
            </bndbox>
        </object>
        ...
    </annotation>

Outputs:
    - Prints the count of objects for each class.
    - Prints the total number of objects in the dataset.

    Expected Output:
        Class Counts: {'with_mask': 3232, 'without_mask': 717, 'mask_weared_incorrect': 123}
        Total Objects (People): 4072 

Usage:
    - Update the `annotation_folder` variable with the path to your annotation directory.
    - Run the script to see the class distribution and total object count.
"""
import os
import xml.etree.ElementTree as ET
from typing import Dict, Tuple


def count_objects_in_dataset(annotation_dir: str) -> Tuple[Dict[str, int], int]:
    """
    Counts the total number of objects (people) in the dataset by parsing all annotation XML files.

    Args:
        annotation_dir (str): Path to the folder containing XML annotation files.

    Returns:
        dict: Total count of objects for each class and the total number of objects.
    """
    class_counts = {}
    total_objects = 0

    for annotation_file in os.listdir(annotation_dir):
        if annotation_file.endswith('.xml'):
            tree = ET.parse(os.path.join(annotation_dir, annotation_file))
            root = tree.getroot()

            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_objects += 1

    return class_counts, total_objects


if __name__ == '__main__':
    # Example usage
    annotation_folder = "./dataset/annotations"  # Replace with your annotation folder path
    class_counts, total_objects = count_objects_in_dataset(annotation_folder)

    print(f"Class Counts: {class_counts}")
    print(f"Total Objects (People): {total_objects}")
