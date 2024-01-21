from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

def create_data_generator(directory, batch_size, img_size, shuffle, augment=False):
    if augment == True:
        datagen = ImageDataGenerator(
                                rotation_range=45,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                vertical_flip=False,
                                brightness_range=[0.5, 1.5],
                                channel_shift_range=0.2,
                                rescale=1./255)
    else: 
        datagen = ImageDataGenerator(
                                rescale=1./255
                                )
    
    generator = datagen.flow_from_directory(
            directory,  # this is the target directory
            target_size=(img_size, img_size),  # all images will be resized to IMG_SIZExIMG_SIZE
            batch_size=batch_size,
            classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
            class_mode='categorical',# since we use binary_crossentropy loss, we need categorical labels
            shuffle=shuffle)  
    
    return generator
    