import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.layers import Embedding, LSTM

# Load pre-trained VGG-16 model without top (classification) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False


# Add layers on top of VGG-16
inputs = Input(shape=(224, 224, 3))
vgg_output = base_model(inputs)
flatten = Flatten()(vgg_output)
dense = Dense(256, activation='relu')(flatten)  # Adjust the output size as needed

max_length = 30
embedding_size = 512
vocabulary_size = 10000

caption_data = 'captions_train.npy'
for input_sequence in caption_data:
    # RNN for caption generation
    embedding = Embedding(vocabulary_size, embedding_size, input_length=max_length)(input_sequence)
    lstm = LSTM(256)(embedding)  # Adjust the LSTM units as needed
    # Merge CNN and RNN
    merged = tf.keras.layers.concatenate([dense, lstm])


output = Dense(vocabulary_size, activation='softmax')(merged)

model = Model(inputs=[inputs, input_sequence], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam')

train_images = 'images_train.npy'
train_captions = 'captions_train.npy'
val_images = 'images_validation.npy'
val_captions = 'captions_validation.npy'
model.fit([train_images, train_captions], train_targets, validation_data=([val_images, val_captions], val_targets), epochs=10, batch_size=64)





