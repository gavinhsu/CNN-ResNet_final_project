#!/usr/bin/env python
# coding: utf-8

# Part1

# In[ ]:


# 資料預處理
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt 
from PIL import Image 
import seaborn as sns
import pandas as pd 
import numpy as np 
import os 

img1 = "cat.1.jpg"
img2 = "cat.9.jpg"
img3 = "dog.23.jpg"
img4 = "dog.31.jpg"
imgs = [img1, img2, img3, img4]

def _load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img 

def _get_predictions(_model):
    f, ax = plt.subplots(1, 4)
    f.set_size_inches(20, 10)
    for i in range(4):
        ax[i].imshow(Image.open(imgs[i]).resize((200, 200), Image.ANTIALIAS))
    plt.show()
    
    f, axes = plt.subplots(1, 4)
    f.set_size_inches(20, 5)
    for i,img_path in enumerate(imgs):
        img = _load_image(img_path)
        preds  = decode_predictions(_model.predict(img), top=3)[0]
        b = sns.barplot(y=[c[1] for c in preds], x=[c[2] for c in preds], color="gray", ax=axes[i])
        b.tick_params(labelsize=8)
        f.tight_layout()


# In[180]:


from keras.applications.resnet50 import ResNet50
resnet_model = ResNet50(weights='imagenet')
_get_predictions(resnet_model)


# Part2

# In[181]:


from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import os

resnet = ResNet50(weights='imagenet', include_top=False)
for layer in resnet.layers[:-7]:
    layer.trainable = False
resnet.summary()


# In[182]:


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
from keras import Model, layers

x = resnet.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.15)(x)
predictions = layers.Dense(3, activation='softmax')(x)
model = Model(resnet.input, predictions)

optimizer = keras.optimizers.Adam(lr=1e-4)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()


# In[183]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    'shapes/train',
    batch_size=32,
    class_mode='binary',
    target_size=(224,224))

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

validation_generator = validation_datagen.flow_from_directory(
    'shapes/validation',
    shuffle=False,
    class_mode='binary',
    target_size=(224,224))


# In[ ]:


import zipfile
from google.colab import drive
drive.mount('/content/drive/')
zip_ref = zipfile.ZipFile("/content/drive/My Drive/Colab Notebooks/basicshapes.zip", 'r')
zip_ref.extractall("/tmp")
zip_ref.close()


# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[184]:


history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=30, # added in Kaggle
                              epochs=8,
                              validation_data=validation_generator,
                              validation_steps=50  # added in Kaggle
                             )


# In[185]:


#Plot Loss and accuracy
import matplotlib.pyplot as plt
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.show()


# In[ ]:


# predict
path = ["cracker.jpg"]
img_list = [Image.open(img_path) for img_path in path]
validation_batch = np.stack([preprocess_input(np.array(img.resize((224,224))))
                             for img in img_list])
pred_probs = model.predict(validation_batch)
pred_probs

