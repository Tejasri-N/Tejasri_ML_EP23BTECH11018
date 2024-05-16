import numpy as np
import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

images = []
captions = []

df = pd.read_csv('captions.csv')

# caption_dict = df.set_index('image').to_dict()
caption_dict = dict(zip(df['image'], df['caption']))

current_dir = 'D:\\ml_project\\grayscaled_images'
for filename in os.listdir(current_dir):
    file_path = os.path.join(current_dir,filename)
    img = Image.open(file_path)
    img_array = (np.array(img))/255.0
    images.append(img_array)
    
    caption = caption_dict.get(filename, '')

    caption_data = [caption]
    # Create a tokenizer and fit it on the captions data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(caption_data)

    # Convert the captions to sequences of token IDs
    sequences = tokenizer.texts_to_sequences(caption_data)

    # Pad or truncate sequences to the specified length
    padded_sequences = pad_sequences(sequences, maxlen=30, padding='post')

    sequence_str = ' '.join(map(str, padded_sequences))

    captions.append(sequence_str)

images = np.array(images)
captions = np.array(captions)
print("Caption preprocessing done....")

images_train, images_test, captions_train, captions_test = train_test_split(images,captions,test_size=0.2,random_state=42)
images_train, images_validation, captions_train, captions_validation = train_test_split(images_train,captions_train,test_size=0.25,random_state=42)

np.save(os.path.join('D:\\ml_project\\','images_train.npy'),images_train)
np.save(os.path.join('D:\\ml_project\\','images_test.npy'),images_test)
np.save(os.path.join('D:\\ml_project\\','images_validation.npy'),images_validation)
np.save(os.path.join('D:\\ml_project\\','captions_train.npy'),captions_train)
np.save(os.path.join('D:\\ml_project\\','captions_test.npy'),captions_test)
np.save(os.path.join('D:\\ml_project\\','captions_validation.npy'),captions_validation)
 
print("Train dataset:")
print("Images dimensions:", images_train.shape)
print("Captions size:", captions_train.shape)
print("\nValidation dataset:")
print("Images dimensions:", images_validation.shape)
print("Captions size:", captions_validation.shape)
print("\nTest dataset:")
print("Images dimensions:", images_test.shape)
print("Captions size:", captions_test.shape)
