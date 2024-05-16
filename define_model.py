import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Embedding, LSTM, concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from New_code import train_sequences, val_sequences

# Load preprocessed data
train_images = np.load('D:\\ml_project\\images_train.npy')
train_captions = np.load('D:\\ml_project\\captions_train.npy')
val_images = np.load('D:\\ml_project\\images_validation.npy')
val_captions = np.load('D:\\ml_project\\captions_validation.npy')

# Parameters
max_length = 30
embedding_size = 512
vocabulary_size = 10000
lstm_units = 256

# Load pre-trained VGG-16 model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Add layers on top of VGG-16
inputs_img = Input(shape=(224, 224, 3))
vgg_output = base_model(inputs_img)
flatten = Flatten()(vgg_output)
dense = Dense(256, activation='relu')(flatten)

# RNN for caption generation
inputs_cap = Input(shape=(max_length,))
embedding = Embedding(vocabulary_size, embedding_size, input_length=max_length)(inputs_cap)
lstm = LSTM(lstm_units)(embedding)

# Merge CNN and RNN outputs
merged = concatenate([dense, lstm])
output = Dense(vocabulary_size, activation='softmax')(merged)

model = Model(inputs=[inputs_img, inputs_cap], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# One-hot encode targets
def one_hot_encode(sequences, vocab_size):
    one_hot = np.zeros((len(sequences), max_length, vocab_size), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, token in enumerate(seq):
            if token < vocab_size:  # Ensure token is within range
                one_hot[i, j, token] = 1.0
    return one_hot

train_targets = one_hot_encode(train_sequences, vocabulary_size)
val_targets = one_hot_encode(val_sequences, vocabulary_size)

# Train the model
model.fit([train_images, train_captions], train_targets,
          validation_data=([val_images, val_captions], val_targets),
          epochs=10, batch_size=64)
