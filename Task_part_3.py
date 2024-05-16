import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# Load captions DataFrame from CSV
df = pd.read_csv('captions.csv')
caption_dict = dict(zip(df['image'], df['caption']))

images = []
captions = []

current_dir = 'D:\\ml_project\\grayscaled_images'

for filename in os.listdir(current_dir):
    file_path = os.path.join(current_dir, filename)
    img = Image.open(file_path)
    img_array = np.array(img) / 255.0
    images.append(img_array)

    caption = caption_dict.get(filename, '')
    captions.append(caption)

images = np.array(images)
captions = np.array(captions)

image_caption_dict = dict(zip(os.listdir(current_dir), captions))

#print(image_caption_dict)

images_train, images_test, captions_train, captions_test = train_test_split(images, captions, test_size=0.2, random_state=50)

images_train, images_val, captions_train, captions_val = train_test_split(images_train, captions_train, test_size=0.25, random_state=50)

print("Train dataset:")
print("Images dimensions:", images_train.shape)
print("Captions size:", captions_train.shape)
print("\nValidation dataset:")
print("Images dimensions:", images_val.shape)
print("Captions size:", captions_val.shape)
print("\nTest dataset:")
print("Images dimensions:", images_test.shape)
print("Captions size:", captions_test.shape)

print(images_train)