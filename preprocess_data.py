import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import os

def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return img.astype('float32')

def preprocess_captions(captions, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences

# Load the datasets
train_data = pd.read_csv('train_data.csv')
val_data = pd.read_csv('val_data.csv')
test_data = pd.read_csv('test_data.csv')

# Tokenize the captions
tokenizer = Tokenizer(num_words=5000, oov_token='<UNK>')
tokenizer.fit_on_texts(train_data['caption'])

max_length = max(len(caption.split()) for caption in train_data['caption'])

# Preprocess the captions
train_captions = preprocess_captions(train_data['caption'], tokenizer, max_length)
val_captions = preprocess_captions(val_data['caption'], tokenizer, max_length)
test_captions = preprocess_captions(test_data['caption'], tokenizer, max_length)

image_dir = 'D:\\ml_project\\Images'

# Function to process images in batches
def process_images(image_paths, batch_size=1000):
    num_images = len(image_paths)
    images = []
    for i in range(0, num_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [load_image(os.path.join(image_dir, image_path)) for image_path in batch_paths]
        images.extend(batch_images)
        print(f"Processed {min(i + batch_size, num_images)} / {num_images} images")
    return np.array(images, dtype='float32')

# Preprocess the images in batches
train_images = process_images(train_data['image'])
val_images = process_images(val_data['image'])
test_images = process_images(test_data['image'])

# Save preprocessed data
np.save('train_images.npy', train_images)
np.save('train_captions.npy', train_captions)
np.save('val_images.npy', val_images)
np.save('val_captions.npy', val_captions)
np.save('test_images.npy', test_images)
np.save('test_captions.npy', test_captions)

# Save the tokenizer and max_length
with open('tokenizer.json', 'w') as f:
    f.write(tokenizer.to_json())
np.save('max_length.npy', max_length)
