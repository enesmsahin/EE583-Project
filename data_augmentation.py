# from keras_preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
#                              shear_range=0.1, zoom_range=0.1, horizontal_flip=True, brightness_range=[0.3, 0.7])


import Augmentor
p = Augmentor.Pipeline("D:/OKUL/6_1/EE583/PROJECT/Dataset/Validation/Metu_red", save_format="jpg")
p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.shear(probability=0.5, max_shear_left=15, max_shear_right=15)
p.flip_left_right(probability=0.5)
p.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.5)

p.sample(100)
