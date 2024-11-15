"""
File: organize_mask_images.py

This script organizes images from a dataset into binary classification folders ('mask' and 'no_mask')
based on their labels extracted from corresponding XML annotation files in PASCAL VOC format.

Modules:
    - os: For directory and file path management.
    - shutil: For copying files to target directories.
    - xml.etree.ElementTree (ET): For parsing XML files to extract annotation data.

Paths:
    - images_path: Directory containing all images.
    - annotations_path: Directory containing XML annotation files.
    - mask_dir: Target directory for images labeled as 'with_mask'.
    - no_mask_dir: Target directory for images labeled as 'without_mask' or 'mask_weared_incorrect'.

Functionality:
    - Iterates through each XML file in the annotations folder.
    - Parses the XML file to find the filename and associated labels.
    - Checks if the label 'with_mask' is present, and moves the corresponding image to the 'mask' folder.
    - Checks if the label 'without_mask' or 'mask_weared_incorrect' is present, and moves the image to the 'no_mask' folder.
    - Ensures that the target directories are created if they do not already exist.

Usage:
    - Adjust the paths 'images_path' and 'annotations_path' as needed to match the directory structure.
    - Run the script in a Python environment to organize images based on their labels.

Assumptions:
    - The images and XML annotation files share the same naming convention (e.g., 'image1.jpg' and 'image1.xml').
    - The XML files contain object labels 'with_mask', 'without_mask', and/or 'mask_weared_incorrect'.
"""
# Import necessary modules
import os
import shutil
import xml.etree.ElementTree as ET

def main():
    # Define relative paths from the current directory
    images_path = './dataset/images' # Directory containing all images
    annotations_path = './dataset/annotations' # Directory containing XML annotation files
    mask_dir = './dataset/binary_dataset/mask' # Target directory for images labeled as 'with_mask'
    no_mask_dir = './dataset/binary_dataset/no_mask' # Target directory for images labeled as 'without_mask' or 'mask_weared_incorrect'

    # Create directories if they don't exist
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(no_mask_dir, exist_ok=True)

    # Iterate over all XML files in the annotations folder
    for xml_file in os.listdir(annotations_path):
        if xml_file.endswith('.xml'):
            # Parse the XML file
            tree = ET.parse(os.path.join(annotations_path, xml_file))
            root = tree.getroot()
            image_file = root.find('filename').text

            # Extract labels from the annotation file
            labels = [obj.find('name').text for obj in root.findall('object')]

            # Classify based on the labels and move the image to the appropriate folder
            if 'with_mask' in labels:
                shutil.copy(os.path.join(images_path, image_file), os.path.join(mask_dir, image_file))
            elif 'without_mask' in labels or 'mask_weared_incorrect' in labels:
                shutil.copy(os.path.join(images_path, image_file), os.path.join(no_mask_dir, image_file))


if __name__ == '__main__':
    # Entry point of the script
    main()
