from PIL import Image
import os

def resize_images(current_folder, destination_folder, desired_size):
    for filename in os.listdir(current_folder):
        file_path = os.path.join(current_folder, filename)  
        with Image.open(file_path) as img:
            resized_img = img.resize(desired_size)
            destination_folder_path = os.path.join(destination_folder, filename)
            resized_img.save(destination_folder_path)
    print("Task finished successfully")

if __name__ == '__main__':

    current_folder = 'D:\\ml_project\\Images'
    destination_folder = 'D:\\ml_project\\resized_images'

    desired_size = (224, 224)  

    resize_images(current_folder, destination_folder, desired_size)