"""
File: organize_mask_images.py

This script processes images and their corresponding XML annotations (in PASCAL VOC format) to crop individual
objects (bounding boxes) and organize the resulting cropped images into class-specific folders for
multi-class classification tasks.

Modules:
    - os: For directory and file path management.
    - xml.etree.ElementTree (ET): For parsing XML files to extract annotation data.
    - PIL.Image: For loading, cropping, and saving image files.

Paths:
    - images_path: Directory containing all images.
    - annotations_path: Directory containing XML annotation files.
    - output_dir: Target directory for storing cropped images, organized into class-specific subfolders
      ('with_mask', 'without_mask', 'mask_weared_incorrect').

Functionality:
    - Iterates through each XML file in the annotations folder.
    - Parses the XML file to extract:
        - Filename of the associated image.
        - Bounding box coordinates for each object.
        - Class labels ('with_mask', 'without_mask', 'mask_weared_incorrect').
    - Crops the bounding box region from the image for each object.
    - Saves the cropped image to the corresponding class folder in the output directory.
    - Ensures that the output directory and class-specific subdirectories are created if they do not already exist.
    - Converts images with an alpha channel (RGBA) to RGB mode to ensure compatibility with JPEG format.

Usage:
    - Update the paths `images_path` and `annotations_path` to match the directory structure of the dataset.
    - Run the script in a Python environment to process and organize the images into the specified format.

Assumptions:
    - The images and XML annotation files share the same naming convention (e.g., 'image1.jpg' and 'image1.xml').
    - The XML files contain valid object labels ('with_mask', 'without_mask', 'mask_weared_incorrect').
    - All bounding boxes are within the dimensions of the associated image.
"""
import os
import xml.etree.ElementTree as ET
from PIL import Image

def main() -> None:
    images_path = './dataset/images'
    annotations_path = './dataset/annotations'
    output_dir = './dataset/cropped_dataset'

    # Class directories
    class_dirs = ['with_mask', 'without_mask', 'mask_weared_incorrect']
    for class_dir in class_dirs:
        os.makedirs(os.path.join(output_dir, class_dir), exist_ok=True)

    # Iterate over XML files
    for xml_file in os.listdir(annotations_path):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(annotations_path, xml_file))
            root = tree.getroot()

            image_file = root.find('filename').text
            image_path = os.path.join(images_path, image_file)
            image = Image.open(image_path)

            # Ensure the image is in RGB mode
            if image.mode == "RGBA":
                image = image.convert("RGB")

            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in class_dirs:
                    continue

                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                # Crop and save image
                cropped_image = image.crop((xmin, ymin, xmax, ymax))
                cropped_image.save(os.path.join(output_dir, class_name, f"{image_file}_{xmin}_{ymin}.jpg"))

if __name__ == '__main__':
    main()
