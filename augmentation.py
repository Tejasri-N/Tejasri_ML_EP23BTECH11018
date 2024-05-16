import os
import numpy as np
from albumentations import (Compose, HorizontalFlip, RandomRotate90, RandomCrop, RandomBrightnessContrast)
from PIL import Image
from albumentations.pytorch import ToTensorV2

# Define input and output folders
input_folder = 'path_to_input_folder'
output_folder = 'path_to_output_folder'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define the data augmentation pipeline
augmentation_pipeline = Compose([
    HorizontalFlip(p=0.5),
    RandomRotate90(p=0.5),
    RandomCrop(width=224, height=224),
    RandomBrightnessContrast(p=0.2),
    ToTensorV2()
])

# Process images in the input folder
for image_name in os.listdir(input_folder):
    # Check if the file is an image
    if image_name.endswith(('.jpg', '.png', '.jpeg')):
        # Load the image using PIL
        image_path = os.path.join(input_folder, image_name)
        image = Image.open(image_path)

        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Apply the augmentation pipeline to the image
        augmented_image = augmentation_pipeline(image=image_array)['image']

        # Convert the augmented image back to a PIL image
        augmented_image_pil = Image.fromarray(np.uint8(augmented_image * 255))

        # Define the path for the augmented image in the output folder
        augmented_image_path = os.path.join(output_folder, 'aug_' + image_name)

        # Save the augmented image
        augmented_image_pil.save(augmented_image_path)

        print(f"Saved augmented image to {augmented_image_path}")

print("Data augmentation completed.")
