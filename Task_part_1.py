# import numpy as np
# import os

# captions = np.load("D:\\ml_project\\captions.txt")
# data = np.array(captions)
# print(data)

import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

images = []
labels = []
current_dir = 'D:\\ml_project\\grayscaled_images'
for filename in os.listdir(current_dir):
    file_path = os.path.join(current_dir,filename)
    img = Image.open(file_path)
    img_array = (np.array(img))/255.0
    images.append(img_array)

images = np.array(images)
print(images)
# images = np.load('D:\\ml_project\\grayscaled_images')
# captions = np.load('D:\\ml_project\\captions.txt')

# images_train, images_test, captions_train, captions_test = train_test_split(images,captions,test_size=0.2,random_state=42)
# images_train, images_validation, captions_train, captions_validation = train_test_split(images_train,captions_train,test_size=0.25,random_state=42)

# np.save('D:\\ml_project\\images_train',images_train)
# np.save('D:\\ml_project\\images_test',images_test)
# np.save('D:\\ml_project\\images_validation',images_validation)
# np.save('D:\\ml_project\\captions_train.txt',captions_train)
# np.save('D:\\ml_project\\captions_test.txt',captions_test)
# np.save('D:\\ml_project\\captions_validation.txt',captions_validation)
